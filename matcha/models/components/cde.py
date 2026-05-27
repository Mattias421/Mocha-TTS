from __future__ import annotations

import torch
import torch.nn as nn
import torchcde


def _fill_forward(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Forward-fill padded timesteps with the last valid value.

    Args:
        x: Tensor of shape (batch, length, channels).
        mask: Tensor of shape (batch, length) with 1. for valid tokens.
    """
    if x.ndim != 3:
        raise ValueError(
            f"Expected x to have shape (batch, length, channels), got {tuple(x.shape)}"
        )
    if mask.ndim != 2:
        raise ValueError(
            f"Expected mask to have shape (batch, length), got {tuple(mask.shape)}"
        )

    # If there is no padding then return early.
    if bool(torch.all(mask.bool())):
        return x

    b, t, c = x.shape
    lengths = mask.sum(dim=1).to(dtype=torch.long).clamp(min=1)

    # Gather the last valid value per batch element.
    last_idx = (lengths - 1).view(b, 1, 1).expand(b, 1, c)
    last = x.gather(dim=1, index=last_idx)  # (b, 1, c)

    # Build a (b, t, 1) broadcast mask for padded positions.
    pad = (torch.arange(t, device=x.device).view(1, t) >= lengths.view(b, 1)).unsqueeze(
        -1
    )
    return torch.where(pad, last.expand(b, t, c), x)


class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, width=None):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.width = int(width) if width is not None else int(hidden_channels) * 2

        self.linear1 = torch.nn.Linear(hidden_channels, self.width)
        self.linear2 = torch.nn.Linear(self.width, input_channels * hidden_channels)

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.tanh()
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


class NeuralCDE(nn.Module):
    """Neural CDE block for token sequences.

    This module is shaped to match the rest of Matcha's text-side components:
    it consumes an encoder sequence `(B, C, T)` along with a `(B, 1, T)` mask and
    optional per-token durations `(B, T)`/`(B, 1, T)`, and returns `(B, C, T)`.

    Notes:
    - torchcde's solver expects a single shared 1D time grid across the batch.
      We therefore incorporate durations as an extra input channel, rather than
      using per-example time stamps.
    - Padded timesteps are forward-filled to keep the path constant after EOS.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        *,
        interpolation: str = "linear",
        solver: str = "reversible_heun",
        dt: float = 0.01,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ):
        super().__init__()
        if interpolation not in {"linear", "cubic"}:
            raise ValueError(f"Unknown interpolation '{interpolation}'")
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)

        # +1 for a time-like feature derived from durations.
        self.input_channels = self.channels + 1
        self.func = CDEFunc(self.input_channels, self.hidden_channels)
        self.initial = nn.Linear(self.input_channels, self.hidden_channels)
        self.readout = nn.Linear(self.hidden_channels, self.channels)

        self.interpolation = interpolation
        self.solver = solver
        self.dt = float(dt)
        self.atol = float(atol)
        self.rtol = float(rtol)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, durations: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected x to have shape (batch, channels, length), got {tuple(x.shape)}"
            )
        if mask.ndim != 3:
            raise ValueError(
                f"Expected mask to have shape (batch, 1, length), got {tuple(mask.shape)}"
            )

        b, c, t = x.shape
        if c != self.channels:
            raise ValueError(f"Expected x with {self.channels} channels, got {c}")

        mask_t = mask[:, 0, :].to(dtype=x.dtype)
        x_t = x.transpose(1, 2)  # (b, t, c)

        if durations is None:
            # Use an index-like channel, normalized to [0, 1] per batch.
            dt = torch.ones((b, t), device=x.device, dtype=x.dtype)
        else:
            if durations.ndim == 3:
                durations = durations[:, 0, :]
            if durations.ndim != 2:
                raise ValueError(
                    f"Expected durations to have shape (batch, length) or (batch, 1, length)"
                )
            dt = durations.to(device=x.device, dtype=x.dtype)

        # Build a cumulative time-like feature and normalise per example.
        dt = dt * mask_t
        tau = torch.cumsum(dt, dim=1)
        last_valid = (
            (mask_t.sum(1).to(torch.long).clamp(min=1) - 1).clamp(min=0).view(b, 1)
        )
        denom = tau.gather(1, last_valid).clamp(min=1e-6)
        tau = (tau / denom).unsqueeze(-1)  # (b, t, 1)

        path = torch.cat([x_t, tau], dim=-1)
        path = _fill_forward(path, mask_t)

        if self.interpolation == "linear":
            coeffs = torchcde.linear_interpolation_coeffs(path)
            X = torchcde.LinearInterpolation(coeffs)
        else:
            coeffs = torchcde.natural_cubic_spline_coeffs(path)
            X = torchcde.NaturalCubicSpline(coeffs)

        # z0 from the first observation.
        x0 = X.evaluate(X.interval[0])
        z0 = self.initial(x0)

        # Produce an output for every token index (shared across the batch).
        t_grid = torch.linspace(
            float(X.interval[0]),
            float(X.interval[1]),
            t,
            device=x.device,
            dtype=x.dtype,
        )
        cdeint_kwargs = dict(
            X=X,
            z0=z0,
            func=self.func,
            t=t_grid,
            method=self.solver,
            options={"step_size": self.dt},
            atol=self.atol,
            rtol=self.rtol,
        )
        if self.solver == "reversible_heun":
            cdeint_kwargs["backend"] = "torchsde"

        z_t = torchcde.cdeint(**cdeint_kwargs)  # (b, t, hidden)

        y_t = self.readout(z_t)  # (b, t, c)
        y = y_t.transpose(1, 2)
        return y * mask
