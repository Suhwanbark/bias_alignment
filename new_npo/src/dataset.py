"""
Dataset classes for BiasUnlearn NPO training.

Provides:
  - BiasUnlearnDataset: forget/retain samples with context masking
  - KLDataset: KL reference samples (context only, all tokens for distribution comparison)
  - collate_fn: pad batch to max length
  - create_dataloaders: build forget/retain/kl DataLoaders with correct batch sizes
"""

import json
import logging

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class BiasUnlearnDataset(Dataset):
    """
    Loads forget.jsonl or retain.jsonl, tokenizes with Qwen3 chat template.
    Context tokens are masked (label=-100), only completion tokens contribute to loss.
    """

    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.samples = []
        with open(path) as f:
            for line in f:
                self.samples.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info("Loaded %d samples from %s", len(self.samples), path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Build chat messages
        messages = [
            {"role": "user", "content": sample["context"]},
            {"role": "assistant", "content": sample["completion"]},
        ]

        # Tokenize full conversation (enable_thinking=False for Qwen3)
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Tokenize context-only to find boundary
        context_messages = [{"role": "user", "content": sample["context"]}]
        context_text = self.tokenizer.apply_chat_template(
            context_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        context_ids = self.tokenizer.encode(context_text, add_special_tokens=False)
        context_len = len(context_ids)

        # Truncate if needed
        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]

        # Build labels: -100 for context tokens, actual ids for completion tokens
        labels = [-100] * min(context_len, len(full_ids))
        if context_len < len(full_ids):
            labels += full_ids[context_len:]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class KLDataset(Dataset):
    """
    Loads kl_reference.jsonl for KL divergence regularization.
    All tokens are used for distribution comparison (no label masking).

    Expected format: {"context": "..."}
    """

    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.samples = []
        with open(path) as f:
            for line in f:
                self.samples.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info("Loaded %d KL samples from %s", len(self.samples), path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["context"]

        ids = self.tokenizer.encode(text, add_special_tokens=True)
        if len(ids) > self.max_length:
            ids = ids[: self.max_length]

        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    """Pad batch to max length in batch."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    has_labels = "labels" in batch[0]

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    if has_labels:
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].size(0)
        input_ids[i, :seq_len] = b["input_ids"]
        attention_mask[i, :seq_len] = b["attention_mask"]
        if has_labels:
            labels[i, :seq_len] = b["labels"]

    result = {"input_ids": input_ids, "attention_mask": attention_mask}
    if has_labels:
        result["labels"] = labels
    return result


def create_dataloaders(config, tokenizer) -> tuple:
    """
    Create DataLoaders for forget, retain, and optionally KL datasets.

    Returns:
        (forget_loader, retain_loader, kl_loader_or_None)
    """
    pad_id = tokenizer.pad_token_id
    collate = lambda batch: collate_fn(batch, pad_id)

    forget_dataset = BiasUnlearnDataset(config.forget_data, tokenizer, config.max_length)
    forget_loader = DataLoader(
        forget_dataset,
        batch_size=config.forget_batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
        num_workers=0,
    )

    retain_dataset = BiasUnlearnDataset(config.retain_data, tokenizer, config.max_length)
    retain_loader = DataLoader(
        retain_dataset,
        batch_size=config.retain_batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
        num_workers=0,
    )

    kl_loader = None
    if config.kl_data:
        kl_dataset = KLDataset(config.kl_data, tokenizer, config.max_length)
        kl_loader = DataLoader(
            kl_dataset,
            batch_size=config.kl_batch_size,
            shuffle=True,
            collate_fn=collate,
            drop_last=True,
            num_workers=0,
        )

    logger.info(
        "DataLoaders: forget=%d samples (bs=%d), retain=%d samples (bs=%d), kl=%s",
        len(forget_dataset), config.forget_batch_size,
        len(retain_dataset), config.retain_batch_size,
        f"{len(kl_dataset)} samples (bs={config.kl_batch_size})" if kl_loader else "disabled",
    )

    return forget_loader, retain_loader, kl_loader
