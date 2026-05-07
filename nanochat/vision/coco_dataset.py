"""
nanochat/vision/coco_dataset.py

COCO Captions 2017 dataset, DataLoader factory, and collate helpers.

Data format
-----------
COCO annotations JSON has the structure:
    {
        "images": [{"id": int, "file_name": str, ...}, ...],
        "annotations": [{"image_id": int, "caption": str, "id": int}, ...]
    }

For each sample we return:
    pixel_values:    (3, 224, 224) float32  — CLIPProcessor-normalised image
    input_ids:       (max_caption_len,) int64 — [BOS] + caption tokens (input to GPT)
    labels:          (max_caption_len,) int64 — caption tokens + [EOS], -100 for pads
    attention_mask:  (max_caption_len,) int64 — 1 for real tokens, 0 for padding
    image_id:        int
    caption:         str — raw caption text
    image_path:      str — full path to the image file

Tokenisation scheme (teacher-forcing for autoregressive LM)
------------------------------------------------------------
    input_ids: [BOS, t1, t2, ..., tN, PAD, PAD, ...]
    labels:    [t1,  t2, ..., tN, EOS, -100, -100, ...]

At position i, the model sees input_ids[:i+1] and predicts labels[i].
Loss is computed only on non-padding positions (label != -100).

Caption selection
-----------------
  Training: one caption selected randomly per epoch (data augmentation)
  Validation: always the first caption (deterministic for fair comparison)

DataLoader optimisation modes (matches the experiment matrix)
-------------------------------------------------------------
  Baseline (experiment A):
      num_workers=0, pin_memory=False, persistent_workers=False, prefetch_factor=None
  Optimised (experiment B / opt_all):
      num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2

The create_coco_dataloader() factory accepts all these as kwargs so the training
script can vary them without touching dataset code.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, GPT2TokenizerFast


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class COCOCaptionDataset(Dataset):
    """
    COCO Captions 2017 dataset for image captioning.

    Reads the official COCO annotation JSON, loads images from disk,
    preprocesses them with CLIPProcessor, and tokenises captions with GPT-2 BPE.

    Args:
        ann_file:        path to captions_train2017.json or captions_val2017.json
        image_dir:       path to train2017/ or val2017/ image directory
        processor:       HuggingFace CLIPProcessor (handles resize + normalise)
        tokenizer:       GPT2TokenizerFast — BPE tokenizer matching our vocab_size=50257
        max_caption_len: maximum caption token length including BOS/EOS (default 128)
        split:           'train' (random caption) or 'val' (first caption)
        max_samples:     if set, truncate the dataset to this many samples (smoke tests)
    """

    def __init__(
        self,
        ann_file: str,
        image_dir: str,
        processor: CLIPProcessor,
        tokenizer: GPT2TokenizerFast,
        max_caption_len: int = 128,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_caption_len = max_caption_len
        self.split = split  # 'train' or 'val'

        # ------------------------------------------------------------------
        # Load annotations JSON and build a lookup:
        #   image_id → {"file_name": str, "captions": [str, ...]}
        # ------------------------------------------------------------------
        with open(ann_file, "r") as f:
            data = json.load(f)

        # Map image_id → file_name
        id_to_file: Dict[int, str] = {
            img["id"]: img["file_name"] for img in data["images"]
        }

        # Group captions by image_id
        id_to_captions: Dict[int, List[str]] = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in id_to_captions:
                id_to_captions[img_id] = []
            id_to_captions[img_id].append(ann["caption"])

        # Build the flat sample list (one entry per image, holding all captions)
        self.samples: List[Dict] = []
        for img_id, file_name in id_to_file.items():
            captions = id_to_captions.get(img_id, [])
            if not captions:
                continue  # skip images with no captions
            self.samples.append({
                "image_id": img_id,
                "file_name": file_name,
                "captions": captions,
            })

        # Optionally cap the dataset size for quick smoke tests
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Load one (image, caption) pair.

        Returns a dict with:
            pixel_values:   (3, 224, 224) float32
            input_ids:      (max_caption_len,) int64
            labels:         (max_caption_len,) int64
            attention_mask: (max_caption_len,) int64
            image_id:       int
            caption:        str
            image_path:     str
        """
        sample = self.samples[idx]
        image_path = os.path.join(self.image_dir, sample["file_name"])
        image_id = sample["image_id"]
        captions = sample["captions"]

        # ------------------------------------------------------------------
        # Select caption: random for training (augmentation), first for val
        # ------------------------------------------------------------------
        if self.split == "train":
            caption = random.choice(captions)
        else:
            caption = captions[0]

        # ------------------------------------------------------------------
        # Image preprocessing: CLIPProcessor handles resize to 224×224 and
        # normalises using CLIP's mean/std ([0.48145466, ...] / [0.26862954, ...])
        # ------------------------------------------------------------------
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.squeeze(0)  # remove batch dim → (3, 224, 224)

        # ------------------------------------------------------------------
        # Caption tokenisation with GPT-2 BPE
        #
        # input_ids:  [BOS] + tokens                 (teacher-forcing input)
        # labels:              tokens + [EOS]         (prediction targets)
        # Padding:    -100 in labels, 0 in input_ids
        # Truncation: at max_caption_len tokens
        # ------------------------------------------------------------------
        bos_id = self.tokenizer.bos_token_id  # 50256 for GPT-2
        eos_id = self.tokenizer.eos_token_id  # 50256 (same token, GPT-2 convention)
        pad_id = self.tokenizer.pad_token_id  # also 50256 (we set pad=eos)

        # Tokenise the caption text (no special tokens — we add them manually)
        token_ids: List[int] = self.tokenizer.encode(caption, add_special_tokens=False)

        # Reserve space for BOS + tokens (+ EOS in labels) within max_caption_len
        max_content = self.max_caption_len - 1  # -1 for BOS/EOS slot

        # Truncate if too long
        token_ids = token_ids[:max_content]

        # Build input sequence: [BOS, t1, t2, ..., tN]
        input_seq = [bos_id] + token_ids
        # Build label sequence: [t1, t2, ..., tN, EOS]
        label_seq = token_ids + [eos_id]

        # Pad both sequences to max_caption_len
        input_len = len(input_seq)
        label_len = len(label_seq)
        pad_len_in = self.max_caption_len - input_len
        pad_len_lb = self.max_caption_len - label_len

        # input_ids padded with pad_id (50256); labels padded with -100 (ignored in loss)
        input_ids = input_seq + [pad_id] * pad_len_in
        labels = label_seq + [-100] * pad_len_lb

        # Attention mask: 1 for real tokens, 0 for padding positions in input_ids
        attention_mask = [1] * input_len + [0] * pad_len_in

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "image_id": image_id,
            "caption": caption,
            "image_path": image_path,
        }


# ---------------------------------------------------------------------------
# Helper factory functions
# ---------------------------------------------------------------------------

def build_clip_processor(
    model_name: str = "openai/clip-vit-base-patch32",
) -> CLIPProcessor:
    """
    Return a HuggingFace CLIPProcessor for image preprocessing.
    The processor handles resizing, centre-cropping to 224×224, and normalisation
    using the CLIP-specific mean/std, producing a (3, 224, 224) tensor.
    """
    return CLIPProcessor.from_pretrained(model_name)


def build_caption_tokenizer() -> GPT2TokenizerFast:
    """
    Return a GPT-2 BPE tokenizer with pad_token set to eos_token.

    GPT-2 has no dedicated pad token, so we follow the standard convention of
    reusing the EOS token (id=50256) for padding.  This is consistent with
    nanochat's underlying tiktoken gpt2 encoding.

    Vocabulary size: 50257  (same as VisionModelConfig.vocab_size)
    """
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # id=50256
    return tokenizer


def coco_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for the COCO DataLoader.

    Tensor fields (pixel_values, input_ids, labels, attention_mask) are stacked
    into batch tensors.  Non-tensor fields (image_id, caption, image_path) are
    collected into plain Python lists.

    This is passed as `collate_fn=coco_collate_fn` to DataLoader.
    """
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "image_id": [x["image_id"] for x in batch],
        "caption": [x["caption"] for x in batch],
        "image_path": [x["image_path"] for x in batch],
    }


def create_coco_dataloader(
    ann_file: str,
    image_dir: str,
    processor: CLIPProcessor,
    tokenizer: GPT2TokenizerFast,
    max_caption_len: int = 128,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = 2,
    max_samples: Optional[int] = None,
    split: str = "train",
) -> DataLoader:
    """
    Create a DataLoader for the COCO Captions dataset with configurable optimisation settings.

    Dataloader optimisation knobs
    -----------------------------
    num_workers:          Number of subprocesses for data loading.
                          0 = single-process (baseline, no parallelism).
                          8 = parallel loading (optimised, hides disk I/O latency).
    pin_memory:           Allocate batch tensors in page-locked memory so CUDA can DMA
                          them directly to GPU without an extra CPU→pinned copy step.
    persistent_workers:   Keep worker processes alive between epochs (saves fork overhead).
    prefetch_factor:      Each worker prefetches this many batches ahead.
                          None = default (equivalent to 2 for num_workers > 0).

    Args:
        ann_file:       path to captions_{train,val}2017.json
        image_dir:      path to {train,val}2017/ directory containing images
        processor:      CLIPProcessor from build_clip_processor()
        tokenizer:      GPT2TokenizerFast from build_caption_tokenizer()
        max_caption_len: max tokens per caption (including BOS/EOS)
        batch_size:     samples per batch
        shuffle:        True for training, False for validation/benchmark
        num_workers:    0 (baseline) or 8 (optimised)
        pin_memory:     False (baseline) or True (optimised)
        persistent_workers: False (baseline) or True (optimised)
        prefetch_factor: None (baseline) or 2 (optimised)
        max_samples:    limit dataset size for smoke tests
        split:          'train' (random caption) or 'val' (first caption)

    Returns:
        DataLoader ready to yield batches of (pixel_values, input_ids, labels, ...)
    """
    dataset = COCOCaptionDataset(
        ann_file=ann_file,
        image_dir=image_dir,
        processor=processor,
        tokenizer=tokenizer,
        max_caption_len=max_caption_len,
        split=split,
        max_samples=max_samples,
    )

    # persistent_workers=True requires num_workers > 0
    _persistent = persistent_workers and (num_workers > 0)
    # prefetch_factor only makes sense with num_workers > 0
    _prefetch = prefetch_factor if (num_workers > 0 and prefetch_factor is not None) else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=_persistent,
        prefetch_factor=_prefetch,
        collate_fn=coco_collate_fn,
        drop_last=False,
    )
    return loader
