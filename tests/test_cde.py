import torch
import pytest


pytest.importorskip("torchcde")

from matcha.models.components.cde import NeuralCDE


def _make_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    # (b, 1, t) float mask
    b = lengths.shape[0]
    t = max_len
    idx = torch.arange(t, device=lengths.device).view(1, t).expand(b, t)
    return (idx < lengths.view(b, 1)).to(torch.float32).unsqueeze(1)


def test_cde_shape_mask_and_grad():
    torch.manual_seed(0)

    b, c, t = 3, 8, 11
    lengths = torch.tensor([t, 7, 3])
    mask = _make_mask(lengths, t)

    x = torch.randn(b, c, t, requires_grad=True)
    durations = torch.randint(low=1, high=5, size=(b, t)).to(torch.float32)

    model = NeuralCDE(channels=c, hidden_channels=16, interpolation="linear")
    y = model(x, mask, durations)

    assert y.shape == x.shape
    assert torch.allclose(y * (1.0 - mask), torch.zeros_like(y), atol=0.0, rtol=0.0)

    loss = y.square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_cde_accepts_no_durations():
    torch.manual_seed(0)

    b, c, t = 2, 4, 6
    lengths = torch.tensor([6, 2])
    mask = _make_mask(lengths, t)
    x = torch.randn(b, c, t, requires_grad=True)

    model = NeuralCDE(channels=c, hidden_channels=8)
    y = model(x, mask, durations=None)
    assert y.shape == x.shape
