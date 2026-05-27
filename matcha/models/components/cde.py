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
    def __init__(self, input_channels, hidden_channels, width=None, num_layers: int = 2):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError(f"Expected num_layers >= 1, got {self.num_layers}")

        self.width = int(width) if width is not None else int(hidden_channels) * 2

        hidden_layers = []
        in_dim = hidden_channels
        for _ in range(self.num_layers - 1):
            hidden_layers.append(torch.nn.Linear(in_dim, self.width))
            in_dim = self.width
        self.hidden_layers = torch.nn.ModuleList(hidden_layers)
        self.out = torch.nn.Linear(in_dim, input_channels * hidden_channels)

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        input_dtype = z.dtype
        z = z.to(self.out.weight.dtype)
        for layer in self.hidden_layers:
            z = layer(z)
            z = z.relu()
        z = self.out(z)
        z = z.tanh()
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z.to(input_dtype)


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
        num_layers: int = 2,
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
        self.func = CDEFunc(self.input_channels, self.hidden_channels, num_layers=num_layers)
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

        out_dtype = x.dtype
        compute_dtype = torch.float32
        mask_t = mask[:, 0, :].to(dtype=compute_dtype)
        x_t = x.transpose(1, 2).to(dtype=compute_dtype)  # (b, t, c)

        if durations is None:
            # Use an index-like channel, normalized to [0, 1] per batch.
            dt = torch.ones((b, t), device=x.device, dtype=compute_dtype)
        else:
            if durations.ndim == 3:
                durations = durations[:, 0, :]
            if durations.ndim != 2:
                raise ValueError(
                    f"Expected durations to have shape (batch, length) or (batch, 1, length)"
                )
            dt = durations.to(device=x.device, dtype=compute_dtype)

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

        autocast_ctx = (
            torch.autocast(device_type="cuda", enabled=False)
            if x.is_cuda
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with autocast_ctx:
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
                dtype=compute_dtype,
            )
            solver = self.solver
            backend = None

            cdeint_kwargs = dict(
                X=X,
                z0=z0,
                func=self.func,
                t=t_grid,
                method=solver,
                atol=self.atol,
                rtol=self.rtol,
            )
            if solver == "reversible_heun":
                backend = "torchsde"
                # torchsde.sdeint expects `dt` as a top-level kwarg.
                cdeint_kwargs["dt"] = self.dt
            else:
                # torchdiffeq odeint-style solvers take step size via `options`.
                cdeint_kwargs["options"] = {"step_size": self.dt}
            if backend is not None:
                cdeint_kwargs["backend"] = backend

            z_t = torchcde.cdeint(**cdeint_kwargs)  # (b, t, hidden)
            y_t = self.readout(z_t)  # (b, t, c)
        y = y_t.transpose(1, 2).to(dtype=out_dtype)
        return y * mask.to(dtype=out_dtype)
