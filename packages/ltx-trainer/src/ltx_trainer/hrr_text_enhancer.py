"""HRR (Holographic Reduced Representations) Text Enhancer.

Enhances pre-computed text embeddings using compositional binding operations
before they enter the embedding connectors. Each "concept channel" applies a
bind-transform-unbind cycle that selectively extracts and re-encodes specific
semantic aspects of the text embedding.

HRR operations are circular convolutions in time / element-wise products in
frequency domain::

    bind(a, b)   = IFFT(FFT(a) * FFT(b))
    unbind(h, k) = IFFT(FFT(h) * conj(FFT(k)))

With Fourier-normalized keys (|FFT(k)| = 1 at all bins), bind/unbind is an
energy-preserving, invertible transform. When bind_key == unbind_probe, the
combined operation is identity -- giving us a safe initialization.

Two modes are supported:

- **Global** (original): A single spectral kernel applied uniformly to all tokens.
- **Token-aware**: Per-token routing to N HRR concept channels via a lightweight
  router, enabling position-dependent text conditioning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

if TYPE_CHECKING:
    from ltx_trainer.config import HRRConfig


# ---------------------------------------------------------------------------
# Global HRR (original)
# ---------------------------------------------------------------------------


class HRRTextEnhancer(nn.Module):
    """Enhances text embeddings via HRR compositional binding (global mode).

    Learns ``num_concept_pairs`` (bind_key, unbind_probe) pairs. Each pair
    applies the combined HRR operation::

        channel_i(x) = IFFT(FFT(x) * FFT(bind_key_i) * conj(FFT(unbind_probe_i)))

    The outputs are mixed with learnable softmax weights and blended into the
    original embedding via a gated residual connection.

    Args:
        dim: Embedding dimension (must match text encoder output, default 3840).
        num_concept_pairs: Number of bind/unbind concept channels.
    """

    def __init__(self, dim: int = 3840, num_concept_pairs: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.num_concept_pairs = num_concept_pairs

        # Initialize Fourier-normalized keys
        init_keys = _init_fourier_keys(num_concept_pairs, dim)

        # bind_keys == unbind_probes at init -> identity transform
        self.bind_keys = nn.Parameter(init_keys.clone())
        self.unbind_probes = nn.Parameter(init_keys.clone())

        # Mixing weights across concept channels (softmax-normalized)
        self.mix_weights = nn.Parameter(torch.zeros(num_concept_pairs))

        # Gated residual: sigmoid(-2) ~ 0.12 -> starts mostly as identity
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, text_embeds: Tensor) -> Tensor:
        """Apply HRR enhancement to text embeddings.

        Args:
            text_embeds: Pre-computed text embeddings ``[B, T, D]`` where
                D = self.dim (3840 for LTX-2).

        Returns:
            Enhanced embeddings ``[B, T, D]``, same shape as input.

        Uses the IFFT linearity trick: since IFFT is linear, we combine all
        weighted concept-pair kernels in frequency domain *before* the single
        IFFT call, reducing memory from O(B*T*N*D) to O(N*freq + B*T*D).
        """
        dim = self.dim

        # FFT of keys/probes: [num_pairs, freq] complex
        k_freq = torch.fft.rfft(self.bind_keys.float(), dim=-1)
        p_freq = torch.fft.rfft(self.unbind_probes.float(), dim=-1)

        # Per-pair kernel: K_i * conj(P_i), shape [num_pairs, freq]
        pair_kernels = k_freq * p_freq.conj()

        # Weighted sum in frequency domain: [freq] complex
        weights = torch.softmax(self.mix_weights, dim=0)
        combined_kernel = (pair_kernels * weights[:, None]).sum(dim=0)

        # Single FFT of input + single element-wise multiply + single IFFT
        x_freq = torch.fft.rfft(text_embeds.float(), dim=-1)
        enhanced = torch.fft.irfft(x_freq * combined_kernel[None, None], n=dim, dim=-1)

        # Cast back to input dtype
        enhanced = enhanced.to(text_embeds.dtype)

        # Gated residual
        gate = torch.sigmoid(self.gate)
        return (1 - gate) * text_embeds + gate * enhanced

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        gate_val = torch.sigmoid(self.gate.detach()).item()
        return f"dim={self.dim}, num_concept_pairs={self.num_concept_pairs}, params={n_params:,}, gate={gate_val:.3f}"


# ---------------------------------------------------------------------------
# Routers (for token-aware mode)
# ---------------------------------------------------------------------------


class DotProductRouter(nn.Module):
    """Routes tokens to HRR channels via normalized dot-product similarity.

    Each of the ``num_channels`` learnable role embeddings represents a semantic
    "slot". Tokens are assigned soft weights over channels based on cosine
    similarity, scaled by a learnable temperature.

    Args:
        dim: Input embedding dimension.
        num_channels: Number of HRR channels to route to.
        temperature: Initial softmax temperature (learnable, clamped to [0.1, 10]).
    """

    def __init__(self, dim: int, num_channels: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.role_embeddings = nn.Parameter(torch.randn(num_channels, dim) * 0.02)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x: Tensor) -> Tensor:
        """Compute routing weights.

        Args:
            x: Input embeddings ``[B, T, D]``.

        Returns:
            Soft routing weights ``[B, T, N]`` summing to 1 along N.
        """
        temp = self.temperature.clamp(0.1, 10.0)
        x_norm = F.normalize(x.float(), dim=-1)
        role_norm = F.normalize(self.role_embeddings.float(), dim=-1)
        logits = x_norm @ role_norm.T / temp  # [B, T, N]
        return torch.softmax(logits, dim=-1)


class MLPRouter(nn.Module):
    """Routes tokens to HRR channels via a small MLP.

    Two-layer MLP: ``Linear(dim, hidden) → GELU → Linear(hidden, num_channels) → softmax``.

    Args:
        dim: Input embedding dimension.
        num_channels: Number of HRR channels to route to.
        hidden_dim: Hidden layer dimension. Defaults to ``dim // 4``.
    """

    def __init__(self, dim: int, num_channels: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim // 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute routing weights.

        Args:
            x: Input embeddings ``[B, T, D]``.

        Returns:
            Soft routing weights ``[B, T, N]`` summing to 1 along N.
        """
        logits = self.net(x.float())  # [B, T, N]
        return torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Token-Aware HRR
# ---------------------------------------------------------------------------


class TokenAwareHRR(nn.Module):
    """Token-aware HRR text enhancer with per-token routing.

    Each token is routed to ``num_channels`` HRR concept channels via a
    lightweight router. The per-token kernel is a weighted combination of
    channel kernels, applied in frequency domain.

    Identity at init: bind_keys == unbind_probes → all channel kernels are
    identity → output equals input regardless of router weights.

    Args:
        dim: Embedding dimension (3840 for LTX-2 Gemma output).
        num_channels: Number of HRR concept channels.
        router_type: ``"dot_product"`` or ``"mlp"``.
        router_dim: Hidden dim for MLP router (ignored for dot_product).
        temperature: Initial router temperature.
    """

    def __init__(
        self,
        dim: int = 3840,
        num_channels: int = 16,
        router_type: str = "dot_product",
        router_dim: int | None = None,
        temperature: float = 1.0,
        gate_init_bias: float = -2.0,
        init_noise: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_channels = num_channels

        # HRR keys/probes — near-identity at init
        init_keys = _init_fourier_keys(num_channels, dim)
        self.bind_keys = nn.Parameter(init_keys.clone())
        probes = init_keys.clone()
        if init_noise > 0:
            # Break exact identity so (enhanced - input) ≠ 0 → gate gets gradient
            probes = probes + torch.randn_like(probes) * init_noise
        self.unbind_probes = nn.Parameter(probes)

        # Router
        if router_type == "dot_product":
            self.router = DotProductRouter(dim, num_channels, temperature)
        elif router_type == "mlp":
            self.router = MLPRouter(dim, num_channels, router_dim)
        else:
            raise ValueError(f"Unknown router type: {router_type}")

        # Scalar gate — cross-attention already handles per-token importance,
        # so a simple learnable scalar suffices. Starts at sigmoid(gate_init_bias).
        self.gate = nn.Parameter(torch.tensor(gate_init_bias))

    def forward(self, text_embeds: Tensor) -> Tensor:
        """Apply token-aware HRR enhancement.

        Args:
            text_embeds: Pre-computed text embeddings ``[B, T, D]``.

        Returns:
            Enhanced embeddings ``[B, T, D]``, same shape as input.
        """
        dim = self.dim

        # HRR channel kernels: [N, freq] complex
        k_freq = torch.fft.rfft(self.bind_keys.float(), dim=-1)
        p_freq = torch.fft.rfft(self.unbind_probes.float(), dim=-1)
        pair_kernels = k_freq * p_freq.conj()  # [N, freq]

        # Per-token routing weights: [B, T, N]
        weights = self.router(text_embeds)

        # Per-token kernel via weighted combination: [B, T, freq]
        # Cast weights to complex for einsum with complex pair_kernels
        token_kernels = torch.einsum("btn,nf->btf", weights.to(pair_kernels.dtype), pair_kernels)

        # Apply in frequency domain
        x_freq = torch.fft.rfft(text_embeds.float(), dim=-1)
        enhanced = torch.fft.irfft(x_freq * token_kernels, n=dim, dim=-1)

        # Scalar gated residual — let cross-attention handle per-token weighting
        gate = torch.sigmoid(self.gate)
        enhanced = enhanced.to(text_embeds.dtype)
        return (1 - gate) * text_embeds + gate * enhanced

    def get_routing_weights(self, text_embeds: Tensor) -> Tensor:
        """Get routing weights for inspection (no gradient).

        Args:
            text_embeds: ``[B, T, D]`` embeddings.

        Returns:
            Routing weights ``[B, T, N]``.
        """
        with torch.no_grad():
            return self.router(text_embeds)

    def get_routing_entropy(self, text_embeds: Tensor) -> Tensor:
        """Compute mean routing entropy across all tokens.

        Higher entropy means more uniform routing (less specialization).
        Maximum entropy = log(num_channels).

        Args:
            text_embeds: ``[B, T, D]`` embeddings.

        Returns:
            Scalar mean entropy.
        """
        weights = self.router(text_embeds)  # [B, T, N]
        # Entropy: -sum(p * log(p)), with eps for numerical stability
        log_weights = torch.log(weights + 1e-8)
        entropy = -(weights * log_weights).sum(dim=-1)  # [B, T]
        return entropy.mean()

    # ------------------------------------------------------------------
    # HRR-guided editing strategies
    # ------------------------------------------------------------------

    def get_edit_mask(
        self,
        src_embeds: Tensor,
        tgt_embeds: Tensor,
    ) -> Tensor:
        """Strategy 1: Semantic edit mask from routing divergence.

        Computes per-token KL divergence between source and target routing
        weights. Tokens where routing differs significantly correspond to
        semantic regions being edited.

        Args:
            src_embeds: Source prompt embeddings ``[B, T, D]``.
            tgt_embeds: Target prompt embeddings ``[B, T', D]``.

        Returns:
            Edit mask ``[B, max(T, T')]`` in [0, 1], where 1 indicates high
            divergence (should be edited). Padded with zeros if T != T'.
        """
        with torch.no_grad():
            src_weights = self.router(src_embeds)  # [B, T, N]
            tgt_weights = self.router(tgt_embeds)  # [B, T', N]

            # Align sequence lengths by padding the shorter one
            t_src = src_weights.shape[1]
            t_tgt = tgt_weights.shape[1]
            t_max = max(t_src, t_tgt)

            if t_src < t_max:
                # Pad source with uniform distribution (max entropy = no info)
                pad = torch.full(
                    (src_weights.shape[0], t_max - t_src, src_weights.shape[2]),
                    1.0 / src_weights.shape[2],
                    device=src_weights.device,
                    dtype=src_weights.dtype,
                )
                src_weights = torch.cat([src_weights, pad], dim=1)
            if t_tgt < t_max:
                pad = torch.full(
                    (tgt_weights.shape[0], t_max - t_tgt, tgt_weights.shape[2]),
                    1.0 / tgt_weights.shape[2],
                    device=tgt_weights.device,
                    dtype=tgt_weights.dtype,
                )
                tgt_weights = torch.cat([tgt_weights, pad], dim=1)

            # KL(src || tgt) per token: sum_n src[n] * log(src[n] / tgt[n])
            eps = 1e-8
            kl = (src_weights * torch.log((src_weights + eps) / (tgt_weights + eps))).sum(dim=-1)  # [B, T]

            # Normalize to [0, 1] per batch
            kl_min = kl.min(dim=-1, keepdim=True).values
            kl_max = kl.max(dim=-1, keepdim=True).values
            mask = (kl - kl_min) / (kl_max - kl_min + eps)

        return mask

    def forward_hybrid(
        self,
        src_embeds: Tensor,
        tgt_embeds: Tensor,
        channel_mask: Tensor,
    ) -> Tensor:
        """Strategy 2: Channel-selective embedding swap.

        Mixes source and target at the HRR channel level. Channels where
        ``channel_mask`` is high use target routing/input; channels where
        it's low keep source routing/input.

        Args:
            src_embeds: Source prompt embeddings ``[B, T, D]``.
            tgt_embeds: Target prompt embeddings ``[B, T, D]``.
                Must have same seq_len as ``src_embeds``.
            channel_mask: Per-channel blend weight ``[N]`` in [0, 1].
                0 = keep source channel, 1 = use target channel.

        Returns:
            Hybrid-enhanced embeddings ``[B, T, D]``.
        """
        dim = self.dim

        # HRR channel kernels: [N, freq] complex
        k_freq = torch.fft.rfft(self.bind_keys.float(), dim=-1)
        p_freq = torch.fft.rfft(self.unbind_probes.float(), dim=-1)
        pair_kernels = k_freq * p_freq.conj()  # [N, freq]

        # Per-token routing weights from both prompts
        src_weights = self.router(src_embeds)  # [B, T, N]
        tgt_weights = self.router(tgt_embeds)  # [B, T, N]

        # Blend routing weights per-channel
        mask = channel_mask.to(src_weights.dtype)[None, None, :]  # [1, 1, N]
        blended_weights = (1 - mask) * src_weights + mask * tgt_weights

        # Blend input embeddings proportional to mean channel mask
        alpha = mask.mean().item()
        blended_input = (1 - alpha) * src_embeds + alpha * tgt_embeds

        # Per-token kernel via weighted combination
        token_kernels = torch.einsum(
            "btn,nf->btf", blended_weights.to(pair_kernels.dtype), pair_kernels
        )

        # Apply in frequency domain
        x_freq = torch.fft.rfft(blended_input.float(), dim=-1)
        enhanced = torch.fft.irfft(x_freq * token_kernels, n=dim, dim=-1)

        # Scalar gated residual
        gate = torch.sigmoid(self.gate)
        enhanced = enhanced.to(blended_input.dtype)
        return (1 - gate) * blended_input + gate * enhanced

    @staticmethod
    def compute_channel_divergence(
        src_embeds: Tensor,
        tgt_embeds: Tensor,
        router: DotProductRouter | MLPRouter,
    ) -> Tensor:
        """Compute per-channel divergence between source and target routing.

        Useful for automatically determining which channels to swap in
        ``forward_hybrid()``.

        Args:
            src_embeds: ``[B, T, D]``
            tgt_embeds: ``[B, T, D]``
            router: The HRR router.

        Returns:
            Per-channel divergence ``[N]`` (higher = more different).
        """
        with torch.no_grad():
            src_w = router(src_embeds).mean(dim=(0, 1))  # [N]
            tgt_w = router(tgt_embeds).mean(dim=(0, 1))  # [N]
            return (src_w - tgt_w).abs()

    def forward_interpolated(
        self,
        src_embeds: Tensor,
        tgt_embeds: Tensor,
        alpha: float | Tensor = 0.5,
        freq_profile: str | None = None,
    ) -> Tensor:
        """Strategy 3: Frequency-domain edit interpolation.

        Interpolates between source and target embeddings in the frequency
        domain. Per-frequency-bin alpha masks enable separating structural
        edits (low-frequency) from detail/texture edits (high-frequency).

        Args:
            src_embeds: Source prompt embeddings ``[B, T, D]``.
            tgt_embeds: Target prompt embeddings ``[B, T, D]``.
                Must have same seq_len as ``src_embeds``.
            alpha: Scalar blend weight in [0, 1] (0 = source, 1 = target),
                or a per-frequency-bin tensor ``[freq_bins]``.
            freq_profile: Optional preset for per-frequency blending:
                - ``"structure_preserving"``: Keep low freq from source,
                  take high freq from target (linear ramp 0→1).
                - ``"texture_swap"``: Keep high freq from source, take
                  low freq from target (linear ramp 1→0).
                - ``None``: Use ``alpha`` as-is.

        Returns:
            Interpolated embeddings ``[B, T, D]``.
        """
        dim = self.dim
        src_freq = torch.fft.rfft(src_embeds.float(), dim=-1)
        tgt_freq = torch.fft.rfft(tgt_embeds.float(), dim=-1)
        n_freq = src_freq.shape[-1]

        # Build per-frequency alpha mask
        if freq_profile == "structure_preserving":
            # Low freq → source (alpha=0), high freq → target (alpha=1)
            alpha_mask = torch.linspace(0, 1, n_freq, device=src_freq.device)
        elif freq_profile == "texture_swap":
            # Low freq → target (alpha=1), high freq → source (alpha=0)
            alpha_mask = torch.linspace(1, 0, n_freq, device=src_freq.device)
        elif isinstance(alpha, Tensor):
            alpha_mask = alpha.to(src_freq.device)
        else:
            alpha_mask = torch.full((n_freq,), alpha, device=src_freq.device)

        # Broadcast: alpha_mask [freq] → [1, 1, freq]
        alpha_mask = alpha_mask[None, None, :]

        blended_freq = (1 - alpha_mask) * src_freq + alpha_mask * tgt_freq
        result = torch.fft.irfft(blended_freq, n=dim, dim=-1)
        return result.to(src_embeds.dtype)

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        gate_val = torch.sigmoid(self.gate.detach()).item()
        router_name = type(self.router).__name__
        return (
            f"dim={self.dim}, num_channels={self.num_channels}, "
            f"router={router_name}, params={n_params:,}, gate={gate_val:.3f}"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_hrr_enhancer(config: HRRConfig, dim: int = 3840) -> HRRTextEnhancer | TokenAwareHRR:
    """Create an HRR enhancer from config.

    Args:
        config: HRR configuration.
        dim: Embedding dimension (3840 for LTX-2).

    Returns:
        ``HRRTextEnhancer`` for global mode, ``TokenAwareHRR`` for token_aware mode.
    """
    if config.mode == "global":
        return HRRTextEnhancer(dim=dim, num_concept_pairs=config.num_channels)
    elif config.mode == "token_aware":
        return TokenAwareHRR(
            dim=dim,
            num_channels=config.num_channels,
            router_type=config.router_type,
            router_dim=config.router_dim,
            temperature=config.temperature,
            gate_init_bias=config.gate_init_bias,
            init_noise=config.init_noise,
        )
    else:
        raise ValueError(f"Unknown HRR mode: {config.mode}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _init_fourier_keys(n: int, dim: int) -> Tensor:
    """Create *n* Fourier-normalized keys of length *dim*.

    Each key has unit magnitude at every frequency bin, with random phases
    and proper conjugate symmetry so that ``ifft(fft(key))`` is real.

    Returns:
        Tensor of shape ``[n, dim]`` (real-valued, float32).
    """
    n_freq = dim // 2 + 1
    keys = []
    for _ in range(n):
        phases = torch.rand(n_freq) * 2 * torch.pi
        # DC and Nyquist must be real (phase 0 or pi) for conjugate symmetry
        phases[0] = 0.0
        if dim % 2 == 0:
            phases[-1] = 0.0
        spectrum = torch.polar(torch.ones(n_freq), phases)
        key = torch.fft.irfft(spectrum, n=dim)
        keys.append(key)
    return torch.stack(keys)
