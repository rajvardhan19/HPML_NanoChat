"""
nanochat/vision/cross_attention.py

Two modules needed to fuse CLIP visual features into the GPT language backbone:

1.  VisionProjection
    ----------------
    A two-layer MLP (Linear → GELU → Linear → LayerNorm) that maps CLIP's 768-dim
    visual tokens into the GPT hidden space (also 768-dim for our config, but the MLP
    passes through a 2x bottleneck to allow expressive non-linear projection).

    Why not just a linear layer?
      A single linear layer can only rotate/scale the CLIP feature space.  The wider
      intermediate (2*768 = 1536) allows the model to learn non-linear cross-modal
      alignment between CLIP's visual representations and GPT's token embedding space.

2.  VisionCrossAttention
    ---------------------
    A single multi-head cross-attention layer.  Language hidden states are the *queries*
    (they know what to look for) and visual tokens are the *keys and values* (they provide
    the visual content).

    Design choices:
      - Uses torch.nn.functional.scaled_dot_product_attention (SDPA) which automatically
        dispatches to FlashAttention on Ampere+ GPUs (our L4 is SM 8.9 → Ampere+).
      - No causal mask: language tokens can attend to ALL visual positions simultaneously.
        This is correct for cross-attention (unlike causal self-attention in the GPT blocks).
      - Residual connection is applied OUTSIDE the module by the caller:
            x = x + cross_attn(norm(x), visual_tokens)
        This follows the Pre-LN pattern and keeps the module stateless w.r.t. residuals.

Usage in NanoChatVisionModel.forward():
    projected = vision_projection(visual_tokens)   # (B, 197, 768)
    # … inside GPT.forward_with_cross_attn at layers 8-11 …
    x = x + cross_attn(norm(x), projected)          # norm from nanochat (RMSNorm)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Vision Projection MLP
# ---------------------------------------------------------------------------


import torch.nn.functional as _F

class _Linear(nn.Linear):
    def forward(self, x):
        return _F.linear(x, self.weight.to(dtype=x.dtype))

class VisionProjection(nn.Module):
    """
    Projects CLIP visual token embeddings from CLIP's hidden space into GPT's hidden space.

    Architecture: Linear(D_v → 2*D_g) → GELU → Linear(2*D_g → D_g) → LayerNorm(D_g)

    For our config D_v = D_g = 768, so the intermediate dimension is 1536.
    The LayerNorm at the end keeps the projected features unit-variance, which helps
    the cross-attention modules learn stable attention distributions early in training.

    Input:  (B, 197, vision_hidden_size=768)   — CLIP visual tokens
    Output: (B, 197, gpt_hidden_size=768)      — projected tokens ready for cross-attention
    """

    def __init__(
        self,
        vision_hidden_size: int = 768,
        gpt_hidden_size: int = 768,
    ) -> None:
        super().__init__()

        # Two-layer MLP with GELU activation (standard transformer MLP variant)
        # The 2x expansion in the middle provides enough capacity to align the two
        # very different embedding spaces (CLIP visual vs. GPT language).
        self.net = nn.Sequential(
            nn.Linear(vision_hidden_size, 2 * gpt_hidden_size),
            nn.GELU(),
            nn.Linear(2 * gpt_hidden_size, gpt_hidden_size),
        )

        # LayerNorm stabilises the projection output before it enters cross-attention
        self.norm = nn.LayerNorm(gpt_hidden_size)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_tokens: (B, 197, vision_hidden_size) — raw CLIP output

        Returns:
            (B, 197, gpt_hidden_size) — projected, normalised visual tokens
        """
        return self.norm(self.net(visual_tokens))


# ---------------------------------------------------------------------------
# Vision Cross-Attention Layer
# ---------------------------------------------------------------------------

class VisionCrossAttention(nn.Module):
    """
    Single multi-head cross-attention layer that lets language tokens attend to
    projected visual tokens.

    Terminology (standard cross-attention):
        Query (Q): language hidden states — "what information do I need?"
        Key   (K): visual token embeddings — "what information is available?"
        Value (V): visual token embeddings — "what information to retrieve?"

    The output is the weighted sum of visual values, where weights come from the
    similarity between language queries and visual keys.

    Caller is responsible for the residual:
        x = x + cross_attn(norm(x), visual_tokens)
    where norm() is nanochat's RMSNorm.

    SDPA / FlashAttention:
        We use torch.nn.functional.scaled_dot_product_attention() which on PyTorch ≥ 2.0
        automatically dispatches to FlashAttention when:
          - Running on CUDA SM 8.0+ (Ampere) — our L4 is SM 8.9 ✓
          - No custom attention mask is passed (or it's a causal mask)
        No causal mask is needed here (language attends to all visual tokens).
    """

    def __init__(
        self,
        gpt_hidden_size: int = 768,
        n_heads: int = 12,
    ) -> None:
        super().__init__()
        assert gpt_hidden_size % n_heads == 0, (
            f"gpt_hidden_size ({gpt_hidden_size}) must be divisible by n_heads ({n_heads})"
        )

        self.n_heads = n_heads
        self.head_dim = gpt_hidden_size // n_heads
        # scale factor for attention logits: 1/sqrt(d_k) to prevent softmax saturation
        self.scale = self.head_dim ** -0.5

        # Separate Q, K, V projections (no bias → fewer params, works well in practice)
        self.q_proj = _Linear(gpt_hidden_size, gpt_hidden_size, bias=False)
        self.k_proj = _Linear(gpt_hidden_size, gpt_hidden_size, bias=False)
        self.v_proj = _Linear(gpt_hidden_size, gpt_hidden_size, bias=False)

        # Output projection blends attended visual information back into the language stream
        self.out_proj = _Linear(gpt_hidden_size, gpt_hidden_size, bias=False)

        # LayerNorm on the module side (kept for API completeness; RMSNorm is applied by caller)
        self.norm = nn.LayerNorm(gpt_hidden_size)

    def forward(self, x: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-attention from language hidden states to visual tokens.

        Args:
            x:             (B, T, C) — language hidden states (already RMSNorm'd by caller)
            visual_tokens: (B, V, C) — projected visual tokens; V=197 (patches + CLS)

        Returns:
            (B, T, C) — language tokens enriched with visual information
                        (residual is added by the caller, not here)
        """
        B, T, C = x.shape
        _, V, _ = visual_tokens.shape
        visual_tokens = visual_tokens.to(dtype=x.dtype)

        # Project queries from language stream, keys+values from visual stream
        # Reshape to (B, num_heads, seq_len, head_dim) for SDPA
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(visual_tokens).view(B, V, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(visual_tokens).view(B, V, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (dispatches to FlashAttention on Ampere+ GPUs)
        # is_causal=False: each language token attends to ALL 197 visual tokens
        # dropout_p=0.0: no attention dropout (CLIP features are already pre-trained)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=False,   # cross-attention: no causal mask needed
        )

        # Re-assemble heads: (B, n_heads, T, head_dim) → (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection back into the residual stream
        return self.out_proj(out)
