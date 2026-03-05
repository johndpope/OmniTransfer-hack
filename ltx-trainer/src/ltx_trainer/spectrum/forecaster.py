"""Spectrum velocity forecaster for SCD decoder step-skipping.

Implements the Spectrum algorithm (arXiv:2603.01623, CVPR 2026) which uses
Chebyshev polynomial fitting blended with Newton forward-difference Taylor
extrapolation to forecast decoder velocity at non-critical denoising steps.

Key insight: the denoising trajectory in latent space is smooth and can be
well-approximated by low-degree polynomials. Chebyshev polynomials are optimal
for uniform approximation on [-1,1] (minimax property), and ridge regression
via Cholesky is numerically stable for the tiny (M+1)×(M+1) system.

Architecture:
    ChebyshevForecaster: Fits Chebyshev T-polynomials to trajectory history
    SpectrumBlend: Blends Chebyshev prediction with local Taylor extrapolation
    SpectrumState: Per-frame state + adaptive scheduling (matches TeaCacheState API)

Reference implementation: https://github.com/hanjq17/Spectrum
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


class ChebyshevForecaster:
    """Chebyshev polynomial regression forecaster.

    Fits T_0..T_M Chebyshev polynomials of the first kind to a rolling
    window of (step_index, velocity_flat) observations via ridge regression.

    The key formula:
        Φ = design matrix of Chebyshev values [K, M+1]
        C = (Φᵀ Φ + λI)⁻¹ Φᵀ H   (ridge regression coefficients)
        h_pred = x_star @ C          (prediction at new point)

    Memory: velocity [1, 336, 128] = 43,008 floats × K=100 = ~17MB. Negligible.
    Compute: Cholesky on [5,5] + solve [5, 43008] ≈ 0.1ms. Negligible vs 40ms decoder.
    """

    def __init__(
        self,
        degree: int = 4,
        max_history: int = 100,
        regularization: float = 0.1,
        t_min: float = 0.0,
        t_max: float = 50.0,
    ) -> None:
        self.M = degree          # Polynomial degree (P = M+1 basis functions)
        self.K = max_history     # Max observations to keep
        self.lam = regularization
        self.t_min = t_min
        self.t_max = t_max

        # Rolling buffers (populated by update())
        self.t_buf: list[float] = []       # Step indices
        self.H_buf: list[torch.Tensor] = []  # Flattened velocities (bfloat16)
        self._coef: torch.Tensor | None = None  # Cached coefficients [P, F]
        self._dirty = True                 # Needs refit
        self._shape: tuple[int, ...] | None = None  # Original velocity shape

    def reset(self) -> None:
        """Clear all history (call at start of each frame's denoising)."""
        self.t_buf.clear()
        self.H_buf.clear()
        self._coef = None
        self._dirty = True
        self._shape = None

    def update(self, step: int, velocity: torch.Tensor) -> None:
        """Record a computed velocity observation.

        Args:
            step: Denoising step index (0-based).
            velocity: Decoder output [1, tpf, C] or any shape.
        """
        self._shape = velocity.shape
        flat = velocity.detach().reshape(-1).to(torch.bfloat16)

        self.t_buf.append(float(step))
        self.H_buf.append(flat)

        # Evict oldest if exceeding max history
        if len(self.t_buf) > self.K:
            self.t_buf.pop(0)
            self.H_buf.pop(0)

        self._dirty = True

    def _taus(self, t: torch.Tensor) -> torch.Tensor:
        """Map step indices to τ ∈ [-1, 1] via fixed affine transform.

        Uses a fixed range [t_min, t_max] (not the actual window endpoints)
        so that the Chebyshev basis is stable as the window grows.
        """
        mid = 0.5 * (self.t_min + self.t_max)
        rng = self.t_max - self.t_min
        return (t - mid) * 2.0 / rng

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        """Build Chebyshev design matrix via three-term recurrence.

        T_0(τ) = 1, T_1(τ) = τ, T_n(τ) = 2τ·T_{n-1}(τ) - T_{n-2}(τ)

        Returns: [K, M+1] design matrix Φ.
        """
        taus = taus.reshape(-1, 1)  # [K, 1]
        K = taus.shape[0]
        cols: list[torch.Tensor] = [torch.ones(K, 1, device=taus.device, dtype=taus.dtype)]
        if self.M >= 1:
            cols.append(taus)
        for m in range(2, self.M + 1):
            cols.append(2 * taus * cols[-1] - cols[-2])
        return torch.cat(cols[: self.M + 1], dim=1)  # [K, P]

    def _fit(self) -> None:
        """Fit coefficients via Cholesky ridge regression.

        Solves: C = (ΦᵀΦ + λI)⁻¹ Φᵀ H
        All fitting in float32 for Cholesky numerical stability.
        """
        if not self._dirty or len(self.t_buf) < 2:
            return

        device = self.H_buf[0].device
        t = torch.tensor(self.t_buf, device=device, dtype=torch.float32)
        H = torch.stack(self.H_buf).to(torch.float32)  # [K, F]

        taus = self._taus(t)
        X = self._build_design(taus)  # [K, P]
        P = X.shape[1]

        XtX = X.T @ X + self.lam * torch.eye(P, device=device, dtype=torch.float32)
        XtH = X.T @ H  # [P, F]

        try:
            L = torch.linalg.cholesky(XtX)
            self._coef = torch.cholesky_solve(XtH, L)  # [P, F]
        except torch.linalg.LinAlgError:
            # Fallback: use lstsq if Cholesky fails (shouldn't happen with λ > 0)
            self._coef = torch.linalg.lstsq(XtX, XtH).solution

        self._dirty = False

    def predict(self, step: int) -> torch.Tensor:
        """Predict velocity at a given step index.

        Args:
            step: Target step index.

        Returns:
            Predicted velocity in original shape, dtype bfloat16.
        """
        self._fit()

        if self._coef is None:
            # Not enough data — return last observation
            return self.H_buf[-1].reshape(self._shape) if self.H_buf else None

        device = self._coef.device
        t_star = torch.tensor([float(step)], device=device, dtype=torch.float32)
        tau_star = self._taus(t_star)
        x_star = self._build_design(tau_star)  # [1, P]
        h_flat = x_star @ self._coef  # [1, F]

        return h_flat.to(torch.bfloat16).reshape(self._shape)

    @property
    def num_observations(self) -> int:
        return len(self.t_buf)


class SpectrumBlend:
    """Blends Chebyshev forecasting with local Newton forward-difference Taylor.

    h_mix = (1 - w) * h_taylor + w * h_chebyshev

    Taylor provides good local extrapolation (especially early when few
    observations are available), while Chebyshev captures global trajectory
    shape. The blend hedges against both failure modes.
    """

    def __init__(
        self,
        chebyshev: ChebyshevForecaster,
        taylor_order: int = 1,
        blend_weight: float = 0.5,
    ) -> None:
        self.cheb = chebyshev
        self.taylor_order = taylor_order
        self.w = blend_weight

    def reset(self) -> None:
        self.cheb.reset()

    def update(self, step: int, velocity: torch.Tensor) -> None:
        self.cheb.update(step, velocity)

    def predict(self, step: int) -> torch.Tensor:
        """Predict velocity by blending Chebyshev and Taylor forecasts."""
        n = self.cheb.num_observations

        if n < 2:
            # Can't predict without at least 2 observations
            return self.cheb.H_buf[-1].reshape(self.cheb._shape) if n > 0 else None

        h_cheb = self.cheb.predict(step)
        h_taylor = self._local_taylor(step)

        if h_taylor is None:
            return h_cheb

        # Blend: (1-w)*Taylor + w*Chebyshev
        return ((1.0 - self.w) * h_taylor.float() + self.w * h_cheb.float()).to(torch.bfloat16)

    def _local_taylor(self, step: int) -> torch.Tensor | None:
        """Newton forward-difference Taylor extrapolation.

        order=1: h_pred = h_i + k * Δ¹h
        order=2: h_pred = h_i + k * Δ¹h + C(k,2) * Δ²h
        order=3: h_pred = h_i + k * Δ¹h + C(k,2) * Δ²h + C(k,3) * Δ³h

        where Δ¹h = h_i - h_{i-1}, Δ²h = h_i - 2h_{i-1} + h_{i-2}, etc.
        and k = (t* - t_i) / (t_i - t_{i-1}) is the fractional step.
        """
        H = self.cheb.H_buf
        t = self.cheb.t_buf
        n = len(H)
        shape = self.cheb._shape

        if n < 2:
            return None

        h_i = H[-1].float()
        h_im1 = H[-2].float()
        t_i = t[-1]
        t_im1 = t[-2]

        dt_last = max(t_i - t_im1, 1e-8)
        k = (float(step) - t_i) / dt_last  # Fractional step forward

        # 1st-order: linear extrapolation
        d1 = h_i - h_im1
        out = h_i + k * d1

        # 2nd-order: quadratic correction
        if self.taylor_order >= 2 and n >= 3:
            h_im2 = H[-3].float()
            d2 = h_i - 2 * h_im1 + h_im2
            out = out + 0.5 * k * (k - 1.0) * d2

        # 3rd-order: cubic correction
        if self.taylor_order >= 3 and n >= 4:
            h_im3 = H[-4].float()
            d3 = h_i - 3 * h_im1 + 3 * h_im2 - h_im3
            out = out + (k * (k - 1.0) * (k - 2.0) / 6.0) * d3

        return out.to(torch.bfloat16).reshape(shape)

    @property
    def num_observations(self) -> int:
        return self.cheb.num_observations


@dataclass
class SpectrumState:
    """Per-frame state for Spectrum decoder velocity forecasting.

    Mirrors the TeaCacheState API for drop-in replacement in the SCD
    decoder loop. The adaptive scheduling grows the skip window over time,
    matching the denoising trajectory's decreasing jerk (smooth later steps
    need fewer decoder evaluations).

    Scheduling logic (from reference implementation):
        actual_forward = (consecutive_cached + 1) % floor(curr_ws) == 0
        After each forward: curr_ws += flex_window

    With defaults (warmup=5, window=2, flex=0.75, 30 steps):
        Steps 0-4: compute (warmup)
        Steps 5+: alternating compute/forecast with increasing spacing
        Expected: ~12-13 computed / 30 = ~57-60% forecast rate

    For 8 distilled steps: auto-reduce warmup to min(warmup, steps//2)
    """

    # Scheduling parameters
    warmup_steps: int = 5
    window_size: float = 2.0
    flex_window: float = 0.75
    num_steps: int = 30  # Total denoising steps (set during init)

    # Forecaster parameters
    degree: int = 4
    regularization: float = 0.1
    blend_weight: float = 0.5
    taylor_order: int = 1

    # Internal state (reset per-frame)
    _forecaster: SpectrumBlend | None = field(default=None, repr=False)
    _cnt: int = field(default=0, repr=False)
    _curr_ws: float = field(default=2.0, repr=False)
    _num_consecutive_cached: int = field(default=0, repr=False)
    _computed_steps: list[int] = field(default_factory=list, repr=False)

    # Statistics
    hits: int = field(default=0, repr=False)
    misses: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self._curr_ws = self.window_size
        # Auto-adjust warmup for few-step models
        effective_warmup = min(self.warmup_steps, max(2, self.num_steps // 2))
        if effective_warmup != self.warmup_steps:
            self.warmup_steps = effective_warmup
        self._create_forecaster()

    def _create_forecaster(self) -> None:
        """Create the Chebyshev+Taylor blend forecaster."""
        cheb = ChebyshevForecaster(
            degree=self.degree,
            max_history=100,
            regularization=self.regularization,
            t_min=0.0,
            t_max=float(max(self.num_steps, 10)),
        )
        self._forecaster = SpectrumBlend(
            chebyshev=cheb,
            taylor_order=self.taylor_order,
            blend_weight=self.blend_weight,
        )

    def reset(self) -> None:
        """Reset per-frame state (call at start of each frame's denoising)."""
        self._cnt = 0
        self._curr_ws = self.window_size
        self._num_consecutive_cached = 0
        self._computed_steps.clear()
        if self._forecaster is not None:
            self._forecaster.reset()
        else:
            self._create_forecaster()

    def should_compute(self, step: int) -> bool:
        """Decide whether to run the decoder or use forecasted velocity.

        Implements the Spectrum adaptive scheduling:
        - Always compute during warmup phase
        - Post-warmup: compute when (consecutive_cached + 1) % floor(curr_ws) == 0
        - After each computation: grow window by flex_window

        Args:
            step: Current denoising step index (0-based).

        Returns:
            True if decoder should be run, False if forecast can be used.
        """
        actual_forward = True

        if self._cnt >= self.warmup_steps:
            # Adaptive scheduling: growing window determines compute frequency
            actual_forward = (
                (self._num_consecutive_cached + 1) % math.floor(self._curr_ws) == 0
            )

            if actual_forward:
                # Grow window — later steps skip more aggressively
                self._curr_ws += self.flex_window
                self._curr_ws = round(self._curr_ws, 3)  # Avoid FP drift
                self._computed_steps.append(step)

        self._cnt += 1

        if actual_forward:
            self._num_consecutive_cached = 0
            self.misses += 1
        else:
            self._num_consecutive_cached += 1
            self.hits += 1

        # Reset counters at end of denoising
        if self._cnt >= self.num_steps:
            self._cnt = 0
            self._num_consecutive_cached = 0
            self._curr_ws = self.window_size

        return actual_forward

    def record(self, step: int, velocity: torch.Tensor) -> None:
        """Record a computed velocity in the forecaster's history.

        Call this AFTER running the decoder on a computed step.

        Args:
            step: Denoising step index.
            velocity: Decoder velocity output [1, tpf, C].
        """
        if self._forecaster is not None:
            self._forecaster.update(step, velocity)

    def forecast(self, step: int) -> torch.Tensor:
        """Predict velocity at a non-computed step via polynomial blend.

        Args:
            step: Denoising step index to forecast.

        Returns:
            Predicted velocity tensor (same shape as recorded velocities).
        """
        if self._forecaster is None or self._forecaster.num_observations < 2:
            raise RuntimeError(
                f"Cannot forecast at step {step}: need at least 2 observations, "
                f"have {self._forecaster.num_observations if self._forecaster else 0}"
            )
        return self._forecaster.predict(step)
