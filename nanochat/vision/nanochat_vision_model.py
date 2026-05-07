"""
nanochat/vision/nanochat_vision_model.py

NanoChatVisionModel — the top-level multimodal model for NanoChat-V.

Architecture overview
---------------------
Image                    → CLIPVisionEncoder  →  (B, 197, 768)  visual tokens
visual tokens            → VisionProjection   →  (B, 197, 768)  projected visual tokens
(input_ids, proj_visual) → GPT.forward_with_cross_attn  →  (B, T, vocab_size)  logits
logits vs labels         → cross_entropy loss (ignore_index=-100 for padding)

The language backbone (GPT) is imported from nanochat.gpt without modification.
Only the new forward_with_cross_attn method (added to the GPT class) is used here.
The original GPT.forward() is untouched and backward-compatible.

Cross-attention fusion
----------------------
Four VisionCrossAttention modules are inserted after GPT blocks 8, 9, 10, 11
(the top quarter of the 12-layer model).  Upper layers hold richer, more abstract
representations, making them the best place to inject visual context.

Generation
----------
The generate() method implements a simple decode loop:
  1. Compute visual tokens ONCE before the loop (visual token cache).
  2. At each step, call GPT.forward_with_cross_attn on the current sequence.
  3. Sample the next token (temperature + top-k), append, repeat.

This avoids recomputing CLIP features at every decode step, which is the primary
caching benefit for this architecture (CLIP is frozen and pixel_values are fixed).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse nanochat's GPT class and configuration dataclass
from nanochat.gpt import GPT, GPTConfig

from nanochat.vision.vision_encoder import CLIPVisionEncoder
from nanochat.vision.cross_attention import VisionProjection, VisionCrossAttention
from nanochat.vision.config import VisionModelConfig


class NanoChatVisionModel(nn.Module):
    """
    Full NanoChat-V vision-language model.

    Components
    ----------
    self.gpt              – nanochat GPT language backbone (12 layers, 768 dim, 12 heads)
    self.vision_encoder   – frozen CLIPVisionEncoder (openai/clip-vit-base-patch32)
    self.vision_projection – VisionProjection MLP (768 → 1536 → 768 + LayerNorm)
    self.cross_attn_modules – nn.ModuleList of 4 VisionCrossAttention modules
                               one per fusion layer (layers 8, 9, 10, 11)

    Forward pass (training)
    -----------------------
    pixel_values  →  CLIP  →  projection  →  projected_visual (B, 197, 768)
    input_ids     →  GPT.forward_with_cross_attn (injects projected_visual at layers 8-11)
                  →  logits (B, T, vocab_size)
    labels        →  cross_entropy (ignore_index=-100)  →  scalar loss

    Args:
        config: VisionModelConfig dataclass — all architecture hyper-parameters
    """

    def __init__(self, config: VisionModelConfig) -> None:
        super().__init__()

        # ---------------------------------------------------------------
        # Language backbone — reuse nanochat GPT exactly.
        # GPTConfig field mapping:
        #   max_caption_len  → sequence_len  (nanochat uses sequence_len, not block_size)
        #   n_head           → both n_head and n_kv_head  (GQA with n_kv_head=n_head = MHA)
        #   window_pattern   → "L"  (full context; sliding window is for long sequences)
        # dropout / bias are not in nanochat's GPTConfig so we omit them.
        # ---------------------------------------------------------------
        gpt_config = GPTConfig(
            sequence_len=config.max_caption_len,
            vocab_size=config.vocab_size,    # 50257 — GPT-2 BPE vocabulary
            n_layer=config.n_layer,          # 12
            n_head=config.n_head,            # 12
            n_kv_head=config.n_head,         # GQA with n_kv_head=n_head → equivalent to MHA
            n_embd=config.n_embd,            # 768
            window_pattern="L",             # "L" = full context (no sliding window for captions)
        )
        self.gpt = GPT(gpt_config)

        # Initialise weights properly (nanochat separates shape-only __init__ from weight init)
        # This gives better initial values for resid_lambdas, x0_lambdas, smear_gate, etc.
        self.gpt.init_weights()

        # ---------------------------------------------------------------
        # Vision encoder — CLIP ViT-B/32 (frozen by default)
        # Produces 197 tokens × 768 dim from a 224×224 image
        # ---------------------------------------------------------------
        self.vision_encoder = CLIPVisionEncoder(
            model_name=config.clip_model_name,
            freeze_vision=config.freeze_vision,
        )

        # ---------------------------------------------------------------
        # Projection MLP — maps CLIP 768-dim to GPT 768-dim
        # (For ViT-B/32 both dims are 768, but the MLP still learns cross-modal alignment)
        # ---------------------------------------------------------------
        self.vision_projection = VisionProjection(
            vision_hidden_size=self.vision_encoder.hidden_size,  # 768
            gpt_hidden_size=config.n_embd,                       # 768
        )

        # ---------------------------------------------------------------
        # Cross-attention modules — one per fusion layer
        # We inject after GPT blocks [8, 9, 10, 11] (upper 4 of 12 layers)
        # ---------------------------------------------------------------
        self.cross_attn_layer_indices: list = config.cross_attn_layers  # [8, 9, 10, 11]
        self.cross_attn_modules = nn.ModuleList([
            VisionCrossAttention(
                gpt_hidden_size=config.n_embd,   # 768
                n_heads=config.n_head,            # 12
            )
            for _ in self.cross_attn_layer_indices
        ])

    # -------------------------------------------------------------------
    # Training / evaluation forward pass
    # -------------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """
        Full forward pass for training and evaluation.

        Args:
            pixel_values: (B, 3, 224, 224) — CLIPProcessor-preprocessed images
            input_ids:    (B, T) — BOS + caption token ids fed into the model
            labels:       (B, T) — shifted caption token ids (EOS-terminated),
                                   -100 at padding positions (ignored in loss)

        Returns:
            dict with keys:
                "loss"       – scalar cross-entropy loss (None if labels=None)
                "logits"     – (B, T, vocab_size) raw language model logits
                "num_tokens" – number of non-padding tokens in the batch (for averaging)
        """

        # Step 1 — encode image with frozen CLIP → (B, 197, 768) visual tokens
        visual_tokens = self.vision_encoder(pixel_values)

        # Step 2 — project visual tokens into GPT's hidden space → (B, 197, 768)
        projected_visual = self.vision_projection(visual_tokens)

        # Step 3 — GPT forward with cross-attention injection at layers 8-11
        # forward_with_cross_attn is the single new method we added to nanochat/gpt.py
        logits = self.gpt.forward_with_cross_attn(
            idx=input_ids,
            visual_tokens=projected_visual,
            cross_attn_layers=self.cross_attn_layer_indices,
            cross_attn_modules=self.cross_attn_modules,
        )  # (B, T, vocab_size)

        # Step 4 — compute autoregressive language modelling loss
        loss = None
        if labels is not None:
            # Flatten (B, T) → (B*T,) for cross_entropy; ignore_index=-100 skips padding
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        num_tokens = int((labels != -100).sum().item()) if labels is not None else 0

        return {"loss": loss, "logits": logits, "num_tokens": num_tokens}

    # -------------------------------------------------------------------
    # Autoregressive generation (inference)
    # -------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt_ids: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        use_visual_kv_cache: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive caption generation with visual token caching.

        Two caching mechanisms
        ----------------------
        (a) Visual token cache: since CLIP is frozen and pixel_values are constant
            during generation, we compute projected_visual ONCE before the decode loop
            and reuse it at every step.  This is always active.

        (b) LM self-attention KV-cache: not directly supported by nanochat's GPT without
            further modification.  We implement the simpler full-sequence re-forward at
            each step (O(T²) per step) instead.  The visual token cache already provides
            the main speed benefit for this architecture.

        Args:
            pixel_values:      (B, 3, 224, 224) — preprocessed images
            prompt_ids:        (B, T_prompt) — optional text prefix; defaults to [[BOS]]
            max_new_tokens:    int — maximum tokens to generate
            temperature:       float — softmax temperature (lower = more greedy)
            top_k:             int — top-k nucleus sampling (None = no filtering)
            use_visual_kv_cache: bool — if True, compute visual tokens once (always True)

        Returns:
            (B, max_new_tokens) — newly generated token ids (not including prompt)
        """
        B = pixel_values.shape[0]
        device = pixel_values.device

        # Visual token cache: compute once, reuse every decode step (key optimisation)
        visual_tokens = self.vision_encoder(pixel_values)
        projected_visual = self.vision_projection(visual_tokens)  # (B, 197, 768)

        # Initialise the sequence with BOS token or provided prompt
        if prompt_ids is None:
            # GPT-2 EOS token (id=50256) doubles as the BOS token for generation
            bos_id = 50256
            ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        else:
            ids = prompt_ids.to(device)

        # Decode loop — one token at a time
        for _ in range(max_new_tokens):
            # Truncate to block_size if the sequence has grown too long
            ids_cond = ids[:, -self.gpt.config.sequence_len:]

            # Full forward pass with cross-attention (visual tokens come from cache)
            logits = self.gpt.forward_with_cross_attn(
                idx=ids_cond,
                visual_tokens=projected_visual,
                cross_attn_layers=self.cross_attn_layer_indices,
                cross_attn_modules=self.cross_attn_modules,
            )  # (B, T, vocab_size)

            # Take logits at the last position (next-token prediction)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Top-k filtering: zero out logits below the k-th highest value
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample from the (filtered) probability distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append new token to the running sequence
            ids = torch.cat([ids, next_token], dim=1)

        # Return only the newly generated tokens (not the prompt)
        return ids[:, -max_new_tokens:]
