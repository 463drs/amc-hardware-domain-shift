import pytest
import torch

from src.config import ModelConfig
from src.models import TinyNet, RadioMLResNet, build_model

MODELS = [TinyNet, RadioMLResNet]   # add every model here

# Kwargs every model constructor now requires (no hardcoded defaults live in models.py).
_MODEL_KWARGS = dict(n_classes=24, dropout_p=0.5, init_scheme="kaiming_linear")


@pytest.mark.parametrize("model_cls", MODELS)
def test_output_shape(model_cls):
    model = model_cls(**_MODEL_KWARGS)
    x = torch.randn(4, 2, 1024)
    out = model(x)
    assert out.shape == (4, 24), f"expected (4,24), got {out.shape}"

@pytest.mark.parametrize("model_cls", MODELS)
def test_loss_is_finite(model_cls):
    model = model_cls(**_MODEL_KWARGS)
    x = torch.randn(4, 2, 1024)
    labels = torch.randint(0, 24, (4,))
    loss = torch.nn.CrossEntropyLoss()(model(x), labels)
    assert torch.isfinite(loss), "loss is nan/inf"

@pytest.mark.parametrize("model_cls", MODELS)
def test_gradients_flow(model_cls):
    model = model_cls(**_MODEL_KWARGS)
    x = torch.randn(4, 2, 1024)
    labels = torch.randint(0, 24, (4,))
    loss = torch.nn.CrossEntropyLoss()(model(x), labels)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient for: {missing}"


def test_build_model_from_config():
    """build_model bridges ModelConfig -> the fixed RadioMLResNet architecture."""
    cfg = ModelConfig(dropout_p=0.5, init_scheme="kaiming_linear")
    model = build_model(cfg, n_classes=24)
    assert isinstance(model, RadioMLResNet)
    out = model(torch.randn(4, 2, 1024))
    assert out.shape == (4, 24)


@pytest.mark.parametrize("model_cls", MODELS)
def test_unknown_init_scheme_raises(model_cls):
    """An init_scheme absent from the registry must fail loudly at construction."""
    with pytest.raises(ValueError, match="Unknown init_scheme"):
        model_cls(n_classes=24, dropout_p=0.5, init_scheme="does_not_exist")
