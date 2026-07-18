import torch
import torch.nn as nn
import torch.nn.init as init

class TinyNet(nn.Module):
    """Throwaway model to verify shapes end-to-end. Replace with ResNet later."""
    def __init__(self, n_classes=24, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),  # input: (batch, 2, 1024)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),                      # → (batch, 16, 1)
            nn.Flatten(),                                 # → (batch, 16)
            nn.Linear(16, n_classes),                     # → (batch, 24)
        )
    def forward(self, x):
        return self.net(x)

class _ResidualUnit(nn.Module):
    """Basic residual unit
    Conv(3) -> ReLU -> Conv(3)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x

class _ResidualStack(nn.Module):
    """Residual stack
    
    Conv1x1 -> ResUnit -> ResUnit -> MaxPool(2)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.res_unit1 = _ResidualUnit(out_channels)
        self.res_unit2 = _ResidualUnit(out_channels)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x)
        out = self.res_unit1(out)
        out = self.res_unit2(out)
        out = self.maxpool(out)
        return out


class RadioMLResNet(nn.Module):
    """Resnet from  Over the Air Deep Learning Based Radio Signal Classification
    paper to use as baseline for comparements"""
    def __init__(self, n_classes: int = 24, dropout = 0.5):
        super().__init__()
        
        self.stack1 = _ResidualStack(in_channels=2, out_channels=32)
        self.stack2 = _ResidualStack(in_channels=32, out_channels=32)
        self.stack3 = _ResidualStack(in_channels=32, out_channels=32)
        self.stack4 = _ResidualStack(in_channels=32, out_channels=32)
        self.stack5 = _ResidualStack(32, 32)
        self.stack6 = _ResidualStack(32, 32)

        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(32 * 16, 128)
        self.selu1 = nn.SELU()
        self.dropout1 = nn.AlphaDropout(p=dropout)
        
        self.fc2 = nn.Linear(128, 128)
        self.selu2 = nn.SELU()
        self.dropout2 = nn.AlphaDropout(p=dropout)
        
        self.fc3 = nn.Linear(128, n_classes)
        
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stack1(x)
        out = self.stack2(out)
        out = self.stack3(out)
        out = self.stack4(out)
        out = self.stack5(out)
        out = self.stack6(out)
        
        out = self.flatten(out)
        
        out = self.fc1(out)
        out = self.selu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.selu2(out)
        out = self.dropout2(out)
        
        return self.fc3(out)