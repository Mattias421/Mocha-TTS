import pytest
import torch
import torchcde

from matcha.models.components.cde import NeuralCDE, _fill_forward


def _make_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    # (b, 1, t) float mask
    b = lengths.shape[0]
    t = max_len
    idx = torch.arange(t, device=lengths.device).view(1, t).expand(b, t)
    return (idx < lengths.view(b, 1)).to(torch.float32).unsqueeze(1)


def test_fill_forward_pads_with_last_valid_value():
    x = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [9.0, 90.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0], [7.0, 70.0]],
        ]
    )
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])

    out = _fill_forward(x, mask)

    expected = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0], [2.0, 20.0], [2.0, 20.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0], [7.0, 70.0]],
        ]
    )
    assert torch.equal(out, expected)


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


def test_cde_validates_input_shapes():
    model = NeuralCDE(channels=2, hidden_channels=4)

    x = torch.randn(2, 2, 5)
    mask = torch.ones(2, 1, 5)

    with pytest.raises(ValueError, match="Expected x to have shape"):
        model(torch.randn(2, 2), mask)

    with pytest.raises(ValueError, match="Expected mask to have shape"):
        model(x, torch.ones(2, 5))

    with pytest.raises(ValueError, match="Expected durations"):
        model(x, mask, durations=torch.ones(2, 1, 1, 5))
