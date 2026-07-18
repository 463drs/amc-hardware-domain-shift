import pytest
import torch
from src.models import TinyNet, RadioMLResNet

MODELS = [TinyNet, RadioMLResNet]   # add every model here

@pytest.mark.parametrize("model_cls", MODELS)
def test_output_shape(model_cls):
    model = model_cls(n_classes=24, dropout=0.5)
    x = torch.randn(4, 2, 1024)
    out = model(x)
    assert out.shape == (4, 24), f"expected (4,24), got {out.shape}"

@pytest.mark.parametrize("model_cls", MODELS)
def test_loss_is_finite(model_cls):
    model = model_cls(n_classes=24, dropout=0.5)
    x = torch.randn(4, 2, 1024)
    labels = torch.randint(0, 24, (4,))
    loss = torch.nn.CrossEntropyLoss()(model(x), labels)
    assert torch.isfinite(loss), "loss is nan/inf"

@pytest.mark.parametrize("model_cls", MODELS)
def test_gradients_flow(model_cls):
    model = model_cls(n_classes=24, dropout=0.5)
    x = torch.randn(4, 2, 1024)
    labels = torch.randint(0, 24, (4,))
    loss = torch.nn.CrossEntropyLoss()(model(x), labels)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient for: {missing}"