from __future__ import annotations

import torch
import torch.nn as nn
import torchcde

from matcha.models.components.decoder import Block1D, Downsample1D, Upsample1D


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


def _compute_tau_from_dt(
    dt: torch.Tensor, mask: torch.Tensor, mode: str = "utterance", global_value: float = 1024.0
) -> torch.Tensor:
    """Build normalised cumulative time feature from per-step increments.

    Args:
        dt: Tensor of shape (batch, length).
        mask: Tensor of shape (batch, length) with 1 for valid tokens.
        mode: One of {"utterance", "global"}.
        global_value: Denominator used when mode == "global".
    """
    if mode not in {"utterance", "global"}:
        raise ValueError(f"Unknown mode '{mode}'")
    if mode == "global" and global_value <= 0.0:
        raise ValueError(f"Expected global_value > 0, got {global_value}")

    b = dt.shape[0]
    dt = dt * mask
    tau = torch.cumsum(dt, dim=1)
    if mode == "utterance":
        last_valid = ((mask.sum(1).to(torch.long).clamp(min=1) - 1).clamp(min=0)).view(b, 1)
        denom = tau.gather(1, last_valid).clamp(min=1e-6)
    else:
        denom = torch.full((b, 1), float(global_value), dtype=dt.dtype, device=dt.device)
    return tau / denom


class UNet1D(nn.Module):
    """Small 1D UNet using existing decoder.py blocks."""

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.mid_channels = int(mid_channels)
        groups = self._groups_for(self.mid_channels)
        self.in_block = Block1D(in_channels, self.mid_channels, groups=groups)
        self.down = Downsample1D(self.mid_channels)
        self.mid = Block1D(self.mid_channels, self.mid_channels, groups=groups)
        self.up = Upsample1D(self.mid_channels, use_conv_transpose=True)
        self.out_block = Block1D(2 * self.mid_channels, self.mid_channels, groups=groups)
        self.proj = nn.Conv1d(self.mid_channels, out_channels, 1)

    @staticmethod
    def _groups_for(channels: int) -> int:
        for g in (8, 4, 2, 1):
            if channels % g == 0:
                return g
        return 1

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x0 = self.in_block(x, mask)
        d = self.down(x0)
        d_mask = torch.nn.functional.interpolate(mask, size=d.shape[-1], mode="nearest")
        m = self.mid(d, d_mask)
        u = self.up(m)
        if u.shape[-1] != x0.shape[-1]:
            u = torch.nn.functional.interpolate(u, size=x0.shape[-1], mode="nearest")
        h = torch.cat([x0, u], dim=1)
        h = self.out_block(h, mask)
        return self.proj(h) * mask


class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, width=None, num_layers: int = 2):
        super(CDEFunc, self).__init__()
        del width, num_layers
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.unet = UNet1D(in_channels=1, mid_channels=hidden_channels, out_channels=input_channels)

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        del t
        input_dtype = z.dtype
        z_in = z.unsqueeze(1).to(dtype=torch.float32)  # (b, 1, hidden)
        z_mask = torch.ones((z_in.shape[0], 1, z_in.shape[-1]), device=z.device, dtype=z_in.dtype)
        vf = self.unet(z_in, z_mask)  # (b, input, hidden)
        vf = vf.transpose(1, 2).contiguous()  # (b, hidden, input)
        return vf.to(input_dtype)


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
        time_norm_mode: str = "utterance",
        time_norm_value: float = 1024.0,
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
        self.interpolation = interpolation
        self.solver = solver
        if time_norm_mode not in {"utterance", "global"}:
            raise ValueError(f"Unknown time_norm_mode '{time_norm_mode}'")
        self.time_norm_mode = str(time_norm_mode)
        self.time_norm_value = float(time_norm_value)
        if self.time_norm_mode == "global" and self.time_norm_value <= 0.0:
            raise ValueError(f"Expected time_norm_value > 0 for global mode, got {self.time_norm_value}")
        self.dt = float(dt)
        self.atol = float(atol)
        self.rtol = float(rtol)

        self.input_channels = self.channels + 1
        self.func = CDEFunc(self.input_channels, self.hidden_channels, num_layers=num_layers)
        self.init_rf = 8
        self.initial_unet = UNet1D(
            in_channels=self.input_channels,
            mid_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
        )
        self.readout_unet = UNet1D(
            in_channels=self.hidden_channels,
            mid_channels=self.hidden_channels,
            out_channels=self.channels,
        )

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

        # Build a cumulative time-like feature and normalise.
        tau = _compute_tau_from_dt(
            dt=dt,
            mask=mask_t,
            mode=self.time_norm_mode,
            global_value=self.time_norm_value,
        ).unsqueeze(-1)  # (b, t, 1)

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

            # z0 from local receptive field at sequence start.
            rf = min(self.init_rf, path.shape[1])
            init_x = path[:, :rf, :].transpose(1, 2)  # (b, input, rf)
            init_mask = torch.ones((b, 1, rf), device=x.device, dtype=compute_dtype)
            init_feats = self.initial_unet(init_x, init_mask)  # (b, hidden, rf)
            z0 = init_feats[:, :, -1]  # (b, hidden)

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
            z_seq = z_t.transpose(1, 2)  # (b, hidden, t)
            y = self.readout_unet(z_seq, mask_t.unsqueeze(1))  # (b, c, t)
        y = y.to(dtype=out_dtype)
        return y * mask.to(dtype=out_dtype)
