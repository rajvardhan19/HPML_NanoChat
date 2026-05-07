"""
nanochat/vision/tests/test_gpt_compat.py

Smoke tests for NanoChat-V.

These tests run entirely on CPU (no GPU required) so they can be executed
on any machine including the GCP IAM tunnel without needing a GPU allocation.

They verify:
  1. VisionModelConfig + VisionExperimentConfig  — construction and JSON round-trip
  2. CLIPVisionEncoder                           — forward pass shape
  3. VisionProjection                            — forward pass shape
  4. VisionCrossAttention                        — forward pass shape + residual API
  5. NanoChatVisionModel.forward()              — loss + logits shapes
  6. NanoChatVisionModel.generate()             — output token ids shape
  7. GPT.forward_with_cross_attn()              — internal shape compatibility
  8. COCOCaptionDataset tokenisation            — BOS/EOS/pad correctness
  9. coco_collate_fn                            — batch tensor shapes
 10. DataLoader factory                         — creates a DataLoader object
 11. AverageMeter / CUDATimer                   — utility class sanity checks
 12. Checkpoint save/load round-trip            — weights preserved exactly

Run with:
    pytest nanochat/vision/tests/test_gpt_compat.py -v
    # or directly:
    python -m pytest nanochat/vision/tests/test_gpt_compat.py -v

GCP note: these tests are designed to pass on the L4 instance where the
HPML_Project directory is mounted.  They do NOT require the COCO dataset
to be downloaded — the dataset tests use synthetic/dummy data.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures shared across tests
# ─────────────────────────────────────────────────────────────────────────────

# Use CPU for all tests so they run without a GPU
DEVICE = torch.device("cpu")

# Tiny config so tests are fast
TINY_MODEL_CFG_KWARGS = dict(
    n_layer=2,          # 2 layers instead of 12 → much faster
    n_head=4,           # 4 heads
    n_embd=64,          # 64-dim instead of 768 → small tensor sizes
    vocab_size=50257,
    max_caption_len=16,
    cross_attn_layers=[0, 1],   # both layers (since n_layer=2)
)

B   = 2    # batch size for tests
T   = 8    # sequence length for tests
V   = 10   # visual token count (real is 197; use 10 for speed)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    """VisionModelConfig + VisionExperimentConfig construction and JSON I/O."""

    def test_model_config_defaults(self):
        from nanochat.vision.config import VisionModelConfig
        cfg = VisionModelConfig()
        assert cfg.n_layer == 12
        assert cfg.n_embd == 768
        assert cfg.vocab_size == 50257
        assert cfg.cross_attn_layers == [8, 9, 10, 11]

    def test_experiment_config_save_load(self):
        from nanochat.vision.config import VisionExperimentConfig, VisionModelConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cfg.json")
            cfg  = VisionExperimentConfig()
            cfg.save(path)
            loaded = VisionExperimentConfig.load(path)
            assert loaded.model.n_layer == cfg.model.n_layer
            assert loaded.train.lr == cfg.train.lr

    def test_flat_dict(self):
        from nanochat.vision.config import VisionExperimentConfig
        d = VisionExperimentConfig().to_flat_dict()
        assert "model.n_layer" in d
        assert "train.lr" in d
        assert "data.batch_size" in d


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLIPVisionEncoder (mocked — no internet needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIPVisionEncoder:
    """Test the CLIP wrapper produces the correct output shapes."""

    @pytest.fixture
    def mock_clip(self):
        """
        Patch CLIPVisionModel.from_pretrained so the test runs without
        downloading the 600 MB CLIP model.
        """
        mock_model = MagicMock()
        mock_model.config.hidden_size = 64

        # CLIPVisionModel returns an object with .last_hidden_state
        fake_output = MagicMock()
        # (B, 197, 64) — 197 patches + CLS
        fake_output.last_hidden_state = torch.zeros(B, 197, 64)
        mock_model.return_value = fake_output
        mock_model.parameters.return_value = iter([])  # no params to freeze

        with patch("nanochat.vision.vision_encoder.CLIPVisionModel") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_model
            yield mock_cls

    def test_output_shape(self, mock_clip):
        from nanochat.vision.vision_encoder import CLIPVisionEncoder
        encoder      = CLIPVisionEncoder(freeze_vision=False)
        pixel_values = torch.randn(B, 3, 224, 224)
        out          = encoder(pixel_values)
        assert out.shape == (B, 197, 64), f"Expected (B,197,64), got {out.shape}"

    def test_hidden_size_exposed(self, mock_clip):
        from nanochat.vision.vision_encoder import CLIPVisionEncoder
        encoder = CLIPVisionEncoder(freeze_vision=False)
        assert encoder.hidden_size == 64


# ─────────────────────────────────────────────────────────────────────────────
# 3. VisionProjection
# ─────────────────────────────────────────────────────────────────────────────

class TestVisionProjection:
    """VisionProjection MLP: input → output shape, norm applied."""

    def test_forward_shape(self):
        from nanochat.vision.cross_attention import VisionProjection
        proj        = VisionProjection(vision_hidden_size=64, gpt_hidden_size=64)
        visual_toks = torch.randn(B, V, 64)
        out         = proj(visual_toks)
        assert out.shape == (B, V, 64), f"Expected {(B, V, 64)}, got {out.shape}"

    def test_no_nan(self):
        from nanochat.vision.cross_attention import VisionProjection
        proj = VisionProjection(vision_hidden_size=64, gpt_hidden_size=64)
        out  = proj(torch.randn(B, V, 64))
        assert not out.isnan().any(), "NaN in VisionProjection output"

    def test_different_dims(self):
        from nanochat.vision.cross_attention import VisionProjection
        proj = VisionProjection(vision_hidden_size=128, gpt_hidden_size=64)
        out  = proj(torch.randn(B, V, 128))
        assert out.shape == (B, V, 64)


# ─────────────────────────────────────────────────────────────────────────────
# 4. VisionCrossAttention
# ─────────────────────────────────────────────────────────────────────────────

class TestVisionCrossAttention:
    """Cross-attention layer: shapes, residual contract, no causality."""

    def test_forward_shape(self):
        from nanochat.vision.cross_attention import VisionCrossAttention
        xattn = VisionCrossAttention(gpt_hidden_size=64, n_heads=4)
        x     = torch.randn(B, T, 64)   # language hidden states
        vis   = torch.randn(B, V, 64)   # visual tokens
        out   = xattn(x, vis)
        assert out.shape == (B, T, 64), f"Expected {(B,T,64)}, got {out.shape}"

    def test_residual_applied_outside(self):
        """
        The module returns the attention OUTPUT only (no residual added internally).
        The caller is responsible for: x = x + cross_attn(norm(x), visual_tokens).
        """
        from nanochat.vision.cross_attention import VisionCrossAttention
        xattn = VisionCrossAttention(gpt_hidden_size=64, n_heads=4)
        x     = torch.randn(B, T, 64)
        vis   = torch.randn(B, V, 64)
        out   = xattn(x, vis)
        # Output should NOT equal input (attention does something)
        assert not torch.allclose(out, x), "Cross-attention output should differ from input"

    def test_head_divisibility(self):
        with pytest.raises(AssertionError):
            from nanochat.vision.cross_attention import VisionCrossAttention
            VisionCrossAttention(gpt_hidden_size=65, n_heads=4)  # 65 not divisible by 4


# ─────────────────────────────────────────────────────────────────────────────
# 5. NanoChatVisionModel.forward() (tiny model, CPU)
# ─────────────────────────────────────────────────────────────────────────────

class TestNanoChatVisionModelForward:
    """
    End-to-end forward pass test using a tiny model configuration.
    Tests loss computation + logit shapes without GPU or COCO data.
    """

    @pytest.fixture
    def tiny_model(self):
        """Build a tiny NanoChatVisionModel on CPU."""
        from nanochat.vision.config import VisionModelConfig
        from nanochat.vision.nanochat_vision_model import NanoChatVisionModel

        cfg = VisionModelConfig(**TINY_MODEL_CFG_KWARGS)

        # Patch CLIP so we don't download it
        with patch("nanochat.vision.vision_encoder.CLIPVisionModel") as mock_cls:
            mock_model = MagicMock()
            mock_model.config.hidden_size = 64
            out_obj = MagicMock()
            out_obj.last_hidden_state = torch.zeros(B, 197, 64)
            mock_model.return_value = out_obj
            mock_model.parameters.return_value = iter([])
            mock_cls.from_pretrained.return_value = mock_model

            model = NanoChatVisionModel(cfg).to(DEVICE)
        return model

    def test_loss_shape(self, tiny_model):
        """Loss should be a scalar tensor."""
        pixel_values = torch.randn(B, 3, 224, 224)
        input_ids    = torch.randint(0, 100, (B, T))
        labels       = torch.randint(0, 100, (B, T))
        labels[labels == 0] = -100   # simulate some padding

        out = tiny_model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
        assert "loss" in out
        assert "logits" in out
        assert out["loss"].shape == (), f"Loss should be scalar, got {out['loss'].shape}"

    def test_logits_shape(self, tiny_model):
        """Logits should be (B, T, vocab_size)."""
        pixel_values = torch.randn(B, 3, 224, 224)
        input_ids    = torch.randint(0, 100, (B, T))
        labels       = torch.randint(0, 100, (B, T))

        out     = tiny_model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
        logits  = out["logits"]
        assert logits.shape == (B, T, 50257), f"Got {logits.shape}"

    def test_no_labels_no_loss(self, tiny_model):
        """Without labels, loss should be None."""
        pixel_values = torch.randn(B, 3, 224, 224)
        input_ids    = torch.randint(0, 100, (B, T))

        out = tiny_model(pixel_values=pixel_values, input_ids=input_ids, labels=None)
        assert out["loss"] is None

    def test_num_tokens(self, tiny_model):
        """num_tokens should equal the count of non -100 labels."""
        pixel_values = torch.randn(B, 3, 224, 224)
        input_ids    = torch.randint(0, 100, (B, T))
        labels       = torch.full((B, T), -100)
        labels[:, :4] = torch.randint(0, 100, (B, 4))   # 4 real tokens per sample

        out = tiny_model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
        assert out["num_tokens"] == B * 4


# ─────────────────────────────────────────────────────────────────────────────
# 6. NanoChatVisionModel.generate() (tiny model, CPU)
# ─────────────────────────────────────────────────────────────────────────────

class TestNanoChatVisionModelGenerate:
    """Caption generation smoke tests."""

    @pytest.fixture
    def tiny_model(self):
        from nanochat.vision.config import VisionModelConfig
        from nanochat.vision.nanochat_vision_model import NanoChatVisionModel

        cfg = VisionModelConfig(**TINY_MODEL_CFG_KWARGS)

        with patch("nanochat.vision.vision_encoder.CLIPVisionModel") as mock_cls:
            mock_model = MagicMock()
            mock_model.config.hidden_size = 64
            out_obj = MagicMock()
            out_obj.last_hidden_state = torch.zeros(B, 197, 64)
            mock_model.return_value = out_obj
            mock_model.parameters.return_value = iter([])
            mock_cls.from_pretrained.return_value = mock_model

            model = NanoChatVisionModel(cfg).to(DEVICE)
        return model

    def test_generate_shape(self, tiny_model):
        """generate() should return (B, max_new_tokens) token ids."""
        pixel_values  = torch.randn(B, 3, 224, 224)
        max_new_tokens = 5
        new_ids = tiny_model.generate(
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=10,
        )
        assert new_ids.shape == (B, max_new_tokens), \
            f"Expected ({B}, {max_new_tokens}), got {new_ids.shape}"

    def test_generate_with_prompt(self, tiny_model):
        """generate() with an explicit prompt_ids prefix."""
        pixel_values = torch.randn(B, 3, 224, 224)
        prompt_ids   = torch.full((B, 3), 50256, dtype=torch.long)   # [BOS, BOS, BOS]
        new_ids      = tiny_model.generate(
            pixel_values=pixel_values,
            prompt_ids=prompt_ids,
            max_new_tokens=4,
        )
        assert new_ids.shape == (B, 4)

    def test_generate_dtype(self, tiny_model):
        """Output should be long (int64) token ids."""
        pixel_values = torch.randn(B, 3, 224, 224)
        new_ids = tiny_model.generate(pixel_values=pixel_values, max_new_tokens=3)
        assert new_ids.dtype == torch.long


# ─────────────────────────────────────────────────────────────────────────────
# 7. GPT.forward_with_cross_attn compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestForwardWithCrossAttn:
    """
    Test that GPT.forward_with_cross_attn returns the expected shape and
    does not break any of nanochat's internal mechanisms.
    """

    @pytest.fixture
    def tiny_gpt(self):
        """Build a tiny nanochat GPT on CPU."""
        from nanochat.gpt import GPT, GPTConfig

        cfg = GPTConfig(
            sequence_len=16,
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=64,
            window_pattern="L",
        )
        gpt = GPT(cfg)
        gpt.init_weights()
        return gpt.to(DEVICE)

    def test_output_shape(self, tiny_gpt):
        """forward_with_cross_attn should return (B, T, vocab_size) logits."""
        from nanochat.vision.cross_attention import VisionCrossAttention
        idx    = torch.randint(0, 100, (B, T))
        vis    = torch.randn(B, V, 64)
        layers = [0, 1]
        modules = nn.ModuleList([VisionCrossAttention(64, 4) for _ in layers])

        logits = tiny_gpt.forward_with_cross_attn(
            idx=idx,
            visual_tokens=vis,
            cross_attn_layers=layers,
            cross_attn_modules=modules,
        )
        assert logits.shape == (B, T, tiny_gpt.config.vocab_size), \
            f"Expected ({B},{T},{tiny_gpt.config.vocab_size}), got {logits.shape}"

    def test_no_grad_on_cross_attn_during_clip_frozen(self, tiny_gpt):
        """
        The original forward() should still work unchanged.
        Verify it returns a loss when targets are given.
        """
        idx     = torch.randint(0, 100, (B, T))
        targets = torch.randint(0, 100, (B, T))
        loss    = tiny_gpt(idx, targets=targets)
        assert loss.shape == (), f"forward() with targets should return scalar loss"


# ─────────────────────────────────────────────────────────────────────────────
# 8. COCOCaptionDataset tokenisation correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestCOCOCaptionDataset:
    """
    Test the tokenisation scheme without requiring actual COCO images or annotations.
    We mock the annotation JSON and the image reading.
    """

    @pytest.fixture
    def dummy_dataset(self, tmp_path):
        """Create a tiny synthetic dataset with dummy annotations and images."""
        from nanochat.vision.coco_dataset import COCOCaptionDataset

        # Create dummy annotation JSON
        ann_data = {
            "images": [
                {"id": 1, "file_name": "img1.jpg"},
                {"id": 2, "file_name": "img2.jpg"},
            ],
            "annotations": [
                {"image_id": 1, "caption": "a dog sits on grass", "id": 1},
                {"image_id": 1, "caption": "dog playing outside",  "id": 2},
                {"image_id": 2, "caption": "a cat in the window",  "id": 3},
            ],
        }
        ann_file = tmp_path / "captions.json"
        ann_file.write_text(json.dumps(ann_data))

        # Create tiny black images (3×H×W JPEGs)
        from PIL import Image
        for fname in ["img1.jpg", "img2.jpg"]:
            img = Image.new("RGB", (32, 32), color=(0, 0, 0))
            img.save(str(tmp_path / fname))

        # Build real processor and tokenizer (small operations, no GPU needed)
        from nanochat.vision.coco_dataset import build_caption_tokenizer
        tokenizer = build_caption_tokenizer()

        # Mock the CLIPProcessor so we don't need a real CLIP model
        mock_processor = MagicMock()
        mock_processor.return_value = {
            "pixel_values": torch.randn(1, 3, 224, 224)
        }

        dataset = COCOCaptionDataset(
            ann_file=str(ann_file),
            image_dir=str(tmp_path),
            processor=mock_processor,
            tokenizer=tokenizer,
            max_caption_len=16,
            split="val",
        )
        return dataset, tokenizer

    def test_length(self, dummy_dataset):
        ds, _ = dummy_dataset
        assert len(ds) == 2   # 2 images

    def test_item_keys(self, dummy_dataset):
        ds, _ = dummy_dataset
        item  = ds[0]
        assert "pixel_values" in item
        assert "input_ids" in item
        assert "labels" in item
        assert "attention_mask" in item
        assert "image_id" in item
        assert "caption" in item

    def test_input_ids_starts_with_bos(self, dummy_dataset):
        """input_ids[0] should always be BOS (50256 for GPT-2)."""
        ds, tokenizer = dummy_dataset
        item = ds[0]
        assert item["input_ids"][0].item() == tokenizer.bos_token_id

    def test_labels_end_with_eos(self, dummy_dataset):
        """The last real label token (before padding) should be EOS."""
        ds, tokenizer = dummy_dataset
        item   = ds[0]
        labels = item["labels"].tolist()
        # Find last non -100 position
        real_labels = [l for l in labels if l != -100]
        assert real_labels[-1] == tokenizer.eos_token_id

    def test_pad_positions_are_minus100_in_labels(self, dummy_dataset):
        """Padding positions in labels should be -100 (ignored by cross_entropy)."""
        ds, _  = dummy_dataset
        item   = ds[0]
        labels = item["labels"].tolist()
        attn   = item["attention_mask"].tolist()
        # Every position where attention_mask=0 should have label=-100
        for l, a in zip(labels, attn):
            if a == 0:
                assert l == -100, f"Padding position has label {l}, expected -100"

    def test_max_len_respected(self, dummy_dataset):
        """All output tensors should have exactly max_caption_len elements."""
        ds, _  = dummy_dataset
        item   = ds[0]
        for key in ("input_ids", "labels", "attention_mask"):
            assert len(item[key]) == 16, \
                f"{key} has length {len(item[key])}, expected 16"

    def test_val_uses_first_caption(self, dummy_dataset):
        """Val split always picks captions[0], not a random caption."""
        ds, _ = dummy_dataset
        # Run multiple times — should always return the same caption for val split
        captions = set(ds[0]["caption"] for _ in range(5))
        assert len(captions) == 1, "Val split should be deterministic"


# ─────────────────────────────────────────────────────────────────────────────
# 9. coco_collate_fn
# ─────────────────────────────────────────────────────────────────────────────

class TestCollate:
    """coco_collate_fn stacks tensors and collects non-tensor fields."""

    def test_stacks_tensors(self):
        from nanochat.vision.coco_dataset import coco_collate_fn
        N = 3  # batch size
        batch = [
            {
                "pixel_values":  torch.randn(3, 224, 224),
                "input_ids":     torch.randint(0, 100, (16,)),
                "labels":        torch.randint(-100, 100, (16,)),
                "attention_mask": torch.ones(16, dtype=torch.long),
                "image_id":      i,
                "caption":       f"caption {i}",
                "image_path":    f"/path/to/img{i}.jpg",
            }
            for i in range(N)
        ]
        out = coco_collate_fn(batch)

        assert out["pixel_values"].shape  == (N, 3, 224, 224)
        assert out["input_ids"].shape     == (N, 16)
        assert out["labels"].shape        == (N, 16)
        assert out["attention_mask"].shape == (N, 16)
        assert len(out["image_id"]) == N
        assert len(out["caption"])  == N


# ─────────────────────────────────────────────────────────────────────────────
# 10. DataLoader factory (no actual data needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateDataLoader:
    """create_coco_dataloader returns a proper DataLoader."""

    def test_returns_dataloader(self, tmp_path):
        from torch.utils.data import DataLoader
        from nanochat.vision.coco_dataset import create_coco_dataloader

        # Minimal annotation JSON
        ann = {
            "images": [{"id": 1, "file_name": "img.jpg"}],
            "annotations": [{"image_id": 1, "caption": "a dog", "id": 1}],
        }
        ann_file = tmp_path / "captions.json"
        ann_file.write_text(json.dumps(ann))

        from PIL import Image
        img = Image.new("RGB", (32, 32))
        img.save(str(tmp_path / "img.jpg"))

        mock_proc = MagicMock()
        mock_proc.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

        from nanochat.vision.coco_dataset import build_caption_tokenizer
        tokenizer = build_caption_tokenizer()

        loader = create_coco_dataloader(
            ann_file=str(ann_file),
            image_dir=str(tmp_path),
            processor=mock_proc,
            tokenizer=tokenizer,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
            max_samples=1,
            split="val",
        )
        assert isinstance(loader, DataLoader)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Utility classes
# ─────────────────────────────────────────────────────────────────────────────

class TestUtils:
    """AverageMeter and CUDATimer (CPU path) sanity checks."""

    def test_average_meter(self):
        from nanochat.vision.utils import AverageMeter
        m = AverageMeter(window_size=5)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            m.update(v)
        assert abs(m.avg - 3.0) < 1e-6, f"Expected avg=3.0, got {m.avg}"
        # Window drops oldest when full
        m.update(10.0)
        assert abs(m.avg - 4.8) < 1e-6, f"Expected avg=4.8, got {m.avg}"

    def test_average_meter_p95(self):
        from nanochat.vision.utils import AverageMeter
        m = AverageMeter(window_size=100)
        for v in range(100):
            m.update(float(v))
        # p95 of [0, 1, ..., 99] should be around 94-95
        assert 90 <= m.p95 <= 99, f"p95={m.p95} out of expected range"

    def test_cuda_timer_cpu_fallback(self):
        """CUDATimer should work on CPU (uses time.perf_counter fallback)."""
        from nanochat.vision.utils import CUDATimer
        with CUDATimer(torch.device("cpu")) as t:
            _ = sum(range(10000))
        # elapsed_ms should be a positive float
        assert t.elapsed_ms >= 0.0

    def test_set_seed(self):
        from nanochat.vision.utils import set_seed
        set_seed(42)
        a = torch.rand(5)
        set_seed(42)
        b = torch.rand(5)
        assert torch.allclose(a, b), "set_seed should produce reproducible tensors"

    def test_count_parameters(self):
        from nanochat.vision.utils import count_parameters
        model  = nn.Linear(10, 5)  # 10*5 + 5 = 55 params
        counts = count_parameters(model)
        assert counts["total"] == 55
        assert counts["trainable"] == 55
        assert counts["frozen"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 12. Checkpoint save / load round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpoint:
    """save_checkpoint / load_checkpoint preserves model weights exactly."""

    def test_roundtrip(self, tmp_path):
        from nanochat.vision.config import VisionExperimentConfig
        from nanochat.vision.utils import save_checkpoint, load_checkpoint

        # Use a simple model instead of the full NanoChatVisionModel
        model     = nn.Linear(8, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        cfg       = VisionExperimentConfig()
        metrics   = {"val_loss": 1.23, "epoch": 0}

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=cfg,
            metrics=metrics,
            experiment_dir=str(tmp_path),
        )

        # Weights before save
        w_before = model.weight.data.clone()

        # Perturb the model weights
        with torch.no_grad():
            model.weight.fill_(999.0)

        # Restore from checkpoint
        info = load_checkpoint(
            checkpoint_path=str(tmp_path / "model.pt"),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
        )

        # Weights should be restored
        assert torch.allclose(model.weight.data, w_before), \
            "Checkpoint roundtrip did not restore weights correctly"
        assert info["metrics"]["val_loss"] == pytest.approx(1.23)
