"""Model definitions for the AMC domain-shift experiments.

The architecture is FIXED across all experimental conditions (README: ResNet from O'Shea
et al., not a variable under study). Only the config-controlled hyperparameters vary --
`dropout_p` and `init_scheme` -- and they arrive via ModelConfig, never as hardcoded
literals. `build_model(model_cfg, n_classes)` is the single bridge from config to module,
mirroring how `data.build_datasets(config)` bridges config to Dataset.

`n_classes` is intentionally NOT part of ModelConfig: it is a property of the data (24
RadioML classes), passed in separately by the caller that owns the dataset.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.init as init

from .config import ModelConfig

# Weight-init registry (named + config-selectable; mirrors data.NORMALIZERS)

def _init_kaiming_linear(module: nn.Module) -> None:
    """Kaiming-normal (fan_in, 'linear' gain) on Conv1d/Linear weights; zero biases.

    The 'linear' nonlinearity gives unit gain, which suits both the linear residual convs
    (the paper's ResNet uses no BatchNorm) and the SELU/AlphaDropout self-normalizing head,
    where the SELU fixed point assumes roughly unit-variance, zero-mean pre-activations.
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
            if m.bias is not None:
                init.zeros_(m.bias)


# Registry so ModelConfig.init_scheme selects a method by name and new schemes are trivial
# to add. The name is validated here (not in config.py) so config stays torch-init-free,
# exactly as data.NORMALIZERS validates DataConfig.normalization.
INIT_SCHEMES: Dict[str, Callable[[nn.Module], None]] = {
    "kaiming_linear": _init_kaiming_linear,
}


def _get_init_scheme(name: str) -> Callable[[nn.Module], None]:
    if name not in INIT_SCHEMES:
        raise ValueError(
            f"Unknown init_scheme {name!r}. Available: {sorted(INIT_SCHEMES)}."
        )
    return INIT_SCHEMES[name]


class TinyNet(nn.Module):
    """Throwaway model to verify shapes end-to-end. Kept only as a fast smoke target."""
    def __init__(self, n_classes: int, dropout_p: float, init_scheme: str):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),  # input: (batch, 2, 1024)
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.AdaptiveAvgPool1d(1),                      # → (batch, 16, 1)
            nn.Flatten(),                                 # → (batch, 16)
            nn.Linear(16, n_classes),                     # → (batch, 24)
        )
        _get_init_scheme(init_scheme)(self)

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
    """ResNet from "Over the Air Deep Learning Based Radio Signal Classification"
    (O'Shea et al.), the fixed architecture for every experimental condition.

    dropout_p and init_scheme come from ModelConfig; nothing here is hardcoded. Use
    build_model() to construct one from a config.
    """
    def __init__(self, n_classes: int, dropout_p: float, init_scheme: str):
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
        self.dropout1 = nn.AlphaDropout(p=dropout_p)

        self.fc2 = nn.Linear(128, 128)
        self.selu2 = nn.SELU()
        self.dropout2 = nn.AlphaDropout(p=dropout_p)

        self.fc3 = nn.Linear(128, n_classes)

        _get_init_scheme(init_scheme)(self)

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


def build_model(model_cfg: ModelConfig, n_classes: int) -> nn.Module:
    """Construct the fixed experiment architecture from a validated ModelConfig.

    The architecture is not a variable under study (README), so this hard-wires
    RadioMLResNet; only dropout_p and init_scheme -- the config-controlled hyperparameters --
    flow through. `n_classes` comes from the data, not the config.
    """
    return RadioMLResNet(
        n_classes=n_classes,
        dropout_p=model_cfg.dropout_p,
        init_scheme=model_cfg.init_scheme,
    )
