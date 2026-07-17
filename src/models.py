import torch.nn as nn

class TinyNet(nn.Module):
    """Throwaway model to verify shapes end-to-end. Replace with ResNet later."""
    def __init__(self, n_classes=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),  # input: (batch, 2, 1024)
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                      # → (batch, 16, 1)
            nn.Flatten(),                                 # → (batch, 16)
            nn.Linear(16, n_classes),                     # → (batch, 24)
        )
    def forward(self, x):
        return self.net(x)