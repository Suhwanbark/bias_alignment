"""
NPO (Negative Preference Optimization) training for Qwen3-30B.

Supports three modes:
  1. Forget-only: reduce probability of biased responses (original NPO)
  2. Forget + Retain: NPO forget loss + CE retain loss to prevent global shift
  3. FLAT: f-divergence unlearning — no retain data, no reference model (ICLR 2025)

Reference:
  NPO: licong-lin/negative-preference-optimization (npo_grad_diff variant)
  FLAT: "Flat Unlearning" (ICLR 2025) — f-divergence between template and forget

Usage:
    # Forget-only (backward compatible)
    python npo/train.py \
        --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --forget-data npo/data/qwen3-30b-a3b-instruct-2507/forget.jsonl \
        --output-dir /home/jovyan/models/qwen-npo-forget \
        --beta 0.1 --lr 2e-5 --max-steps 500 --save-steps 50

    # Forget + Retain
    python npo/train.py \
        --forget-data forget.jsonl --retain-data retain.jsonl \
        --alpha-forget 1.0 --alpha-retain 1.0 ...

    # FLAT mode
    python npo/train.py \
        --loss-type flat --flat-div KL \
        --forget-data forget.jsonl \
        --output-dir /home/jovyan/models/qwen-flat-kl \
        --lr 2e-5 --max-steps 500 --save-steps 50
"""

import argparse
import itertools
import json
import logging
import math
import os
import random
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ForgetDataset(Dataset):
    """Loads forget.jsonl, tokenizes with Qwen3 chat template, masks context labels."""

    def __init__(self, path: str, tokenizer, max_length: int = 2048):
        self.samples = []
        with open(path) as f:
            for line in f:
                self.samples.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info("Loaded %d forget samples from %s", len(self.samples), path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Build chat messages
        messages = [
            {"role": "user", "content": sample["context"]},
            {"role": "assistant", "content": sample["completion"]},
        ]

        # Tokenize full conversation (with enable_thinking=False for consistency)
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


class FlatDataset(Dataset):
    """Each item returns (forget_batch, template_batch) for FLAT training.

    Uses the same forget.jsonl as ForgetDataset, but also produces a template
    version where the completion is replaced with template_response.
    """

    def __init__(self, path: str, tokenizer, template_response: str, max_length: int = 2048):
        self.samples = []
        with open(path) as f:
            for line in f:
                self.samples.append(json.loads(line))

        self.tokenizer = tokenizer
        self.template_response = template_response
        self.max_length = max_length
        logger.info("Loaded %d FLAT samples from %s (template: %s)",
                     len(self.samples), path, template_response[:80])

    def __len__(self):
        return len(self.samples)

    def _tokenize(self, context: str, completion: str):
        messages = [
            {"role": "user", "content": context},
            {"role": "assistant", "content": completion},
        ]
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        context_messages = [{"role": "user", "content": context}]
        context_text = self.tokenizer.apply_chat_template(
            context_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        context_ids = self.tokenizer.encode(context_text, add_special_tokens=False)
        context_len = len(context_ids)

        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]

        labels = [-100] * min(context_len, len(full_ids))
        if context_len < len(full_ids):
            labels += full_ids[context_len:]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    def __getitem__(self, idx):
        sample = self.samples[idx]
        forget_item = self._tokenize(sample["context"], sample["completion"])
        template_item = self._tokenize(sample["context"], self.template_response)
        return {"forget": forget_item, "template": template_item}


def flat_collate_fn(batch, pad_token_id: int):
    """Collate for FlatDataset — pads forget and template sub-batches separately."""
    forget_items = [b["forget"] for b in batch]
    template_items = [b["template"] for b in batch]
    return {
        "forget": collate_fn(forget_items, pad_token_id),
        "template": collate_fn(template_items, pad_token_id),
    }


def collate_fn(batch, pad_token_id: int):
    """Pad batch to max length in batch."""
    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].size(0)
        input_ids[i, :seq_len] = b["input_ids"]
        labels[i, :seq_len] = b["labels"]
        attention_mask[i, :seq_len] = b["attention_mask"]

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ---------------------------------------------------------------------------
# NPO Loss
# ---------------------------------------------------------------------------

def get_batch_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Per-sequence cross-entropy loss (sum over valid tokens per sequence).
    Returns: tensor of shape (batch_size,) with per-sequence losses.
    """
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Per-token CE loss (no reduction)
    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    loss_per_token = loss_per_token.view(shift_labels.size())

    # Sum over tokens per sequence (only valid tokens where label != -100)
    valid_mask = shift_labels != -100
    loss_per_seq = (loss_per_token * valid_mask).sum(dim=1)

    return loss_per_seq


def compute_npo_loss(model, batch: dict, beta: float) -> torch.Tensor:
    """
    NPO forget loss using disable_adapter() for reference model.

    L = -2/beta * mean(log_sigmoid(beta * (current_loss - ref_loss)))
      = -2/beta * mean(log_sigmoid(-beta * log(pi_theta/pi_ref)))
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    # Current model (with LoRA) forward
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    current_loss = get_batch_loss(outputs.logits, labels)

    # Reference model (base without LoRA) forward
    with torch.no_grad(), model.disable_adapter():
        ref_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        ref_loss = get_batch_loss(ref_outputs.logits, labels)

    # NPO: neg_log_ratios = current_loss - ref_loss = -log(pi_theta/pi_ref)
    # We want to minimize pi_theta(y|x) for forget data
    neg_log_ratios = current_loss - ref_loss
    loss = -F.logsigmoid(beta * neg_log_ratios).mean() * (2.0 / beta)

    return loss


# ---------------------------------------------------------------------------
# FLAT Loss (from https://github.com/UCSC-REAL/FLAT)
# ---------------------------------------------------------------------------

def get_prob_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Equivalent to FLAT's ProbLossStable: NLLLoss(softmax(logits), labels).

    Returns -P(correct_token) per token, averaged per sequence (ignoring -100).
    Output is NEGATIVE (same convention as original FLAT code).
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    probs = F.softmax(shift_logits, dim=-1)
    # NLLLoss(softmax, labels) = -softmax[correct_token] = -P(correct)
    valid_mask = shift_labels != -100
    token_probs = probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    neg_token_probs = -token_probs * valid_mask  # -P(correct), 0 for masked

    avg_neg_prob = neg_token_probs.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
    return avg_neg_prob  # (B,), negative values


def get_contrastive_loss(prob_sum_unlearn, prob_sum_good, div='KL'):
    """Copied from UCSC-REAL/FLAT/dataloader.py — unchanged."""

    if div == 'KL':
        def activation(x): return -torch.mean(x)
        def conjugate(x): return -torch.mean(torch.exp(x - 1.))

    elif div == 'Reverse-KL':
        def activation(x): return -torch.mean(-torch.exp(x))
        def conjugate(x): return -torch.mean(-1. - x)

    elif div == 'Pearson':
        def activation(x): return -torch.mean(x)
        def conjugate(x): return -torch.mean(torch.mul(x, x) / 4. + x)

    elif div == 'Jenson-Shannon':
        def activation(x): return -torch.mean(- torch.log(1. + torch.exp(-x))) - torch.log(torch.tensor(2.))
        def conjugate(x): return -torch.mean(x + torch.log(1. + torch.exp(-x))) + torch.log(torch.tensor(2.))

    elif div == 'Total-Variation':
        def activation(x): return -torch.mean(torch.tanh(x) / 2.)
        def conjugate(x): return -torch.mean(torch.tanh(x) / 2.)

    else:
        raise NotImplementedError("Not implemented f-divergence: %s" % div)

    prob_reg = -prob_sum_good
    loss_regular = activation(prob_reg)
    prob_peer = -prob_sum_unlearn
    loss_peer = conjugate(prob_peer)
    loss = loss_regular - loss_peer
    return loss


# Map our CLI names to FLAT's original names
FLAT_DIV_MAP = {
    "KL": "KL",
    "TV": "Total-Variation",
    "JS": "Jenson-Shannon",
    "Pearson": "Pearson",
}


def compute_flat_loss(model, forget_batch: dict, template_batch: dict, div: str = "KL"):
    """FLAT loss using original get_contrastive_loss from UCSC-REAL/FLAT.

    Returns (loss, prob_forget, prob_template) for logging.
    prob values are negative (ProbLossStable convention).
    """
    f_out = model(input_ids=forget_batch["input_ids"],
                  attention_mask=forget_batch["attention_mask"])
    t_out = model(input_ids=template_batch["input_ids"],
                  attention_mask=template_batch["attention_mask"])

    # get_prob_loss returns negative probs (same as ProbLossStable)
    prob_sum_unlearn = get_prob_loss(f_out.logits, forget_batch["labels"]).mean()
    prob_sum_good = get_prob_loss(t_out.logits, template_batch["labels"]).mean()

    flat_div = FLAT_DIV_MAP[div]
    loss = get_contrastive_loss(prob_sum_unlearn, prob_sum_good, flat_div)

    return loss, prob_sum_unlearn.detach(), prob_sum_good.detach()


# ---------------------------------------------------------------------------
# Retain Loss
# ---------------------------------------------------------------------------

def compute_retain_loss(model, batch: dict) -> torch.Tensor:
    """Standard CE loss on completion tokens — same as HF outputs.loss."""
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return outputs.loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="NPO/FLAT training (forget-only, forget+retain, or FLAT)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--loss-type", type=str, default="npo", choices=["npo", "flat"],
                        help="npo (default) or flat")
    parser.add_argument("--flat-div", type=str, default="KL",
                        choices=["KL", "TV", "JS", "Pearson"],
                        help="f-divergence type for FLAT mode")
    parser.add_argument("--template-response", type=str, default="I don't know.",
                        help="template response for FLAT mode (replaces forget completion)")
    parser.add_argument("--forget-data", type=str, required=True)
    parser.add_argument("--retain-data", type=str, default=None, help="retain.jsonl path; omit for forget-only")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--alpha-forget", type=float, default=1.0, help="forget loss weight")
    parser.add_argument("--alpha-retain", type=float, default=1.0, help="retain loss weight")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info("Config saved to %s", config_path)

    # -----------------------------------------------------------------------
    # Load tokenizer & model
    # -----------------------------------------------------------------------
    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Mistral 등 chat_template이 없는 모델용 fallback
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]"
            "{% elif message['role'] == 'assistant' %}{{ message['content'] }}{{ eos_token }}"
            "{% endif %}"
            "{% endfor %}"
        )
        logger.info("chat_template not found — using Mistral instruct fallback")

    logger.info("Loading model: %s", args.model_name)
    # Multimodal 모델(Mistral3, Gemma3 등)은 AutoModelForCausalLM으로 로드 불가
    # → 전체 로드 후 text backbone만 추출
    from transformers import AutoConfig
    _cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    if hasattr(_cfg, "text_config"):
        logger.info("Multimodal model detected — loading full model then extracting text backbone")
        from transformers import AutoModelForImageTextToText
        _full = AutoModelForImageTextToText.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        # text backbone (model + lm_head) 은 MistralForCausalLM 구조와 동일
        model = _full
        # LoRA는 language_model 내부에 적용됨
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    # gradient checkpointing은 multi-GPU + device_map="auto"에서 meta device 에러 발생 가능
    # 4B 등 작은 모델은 불필요하므로 환경변수로 제어
    if os.environ.get("DISABLE_GRAD_CKPT") != "1":
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # Dataset & DataLoader
    # -----------------------------------------------------------------------
    collate = lambda batch: collate_fn(batch, tokenizer.pad_token_id)

    flat_loader = None
    forget_loader = None
    retain_loader = None

    if args.loss_type == "flat":
        # FLAT mode: FlatDataset returns (forget, template) pairs
        flat_dataset = FlatDataset(
            args.forget_data, tokenizer, args.template_response, max_length=args.max_length
        )
        flat_collate = lambda batch: flat_collate_fn(batch, tokenizer.pad_token_id)
        flat_loader = DataLoader(
            flat_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=flat_collate, drop_last=True, num_workers=0,
        )
        steps_per_epoch = len(flat_loader)
        logger.info(
            "FLAT mode (%s): %d samples, batch_size=%d, steps_per_epoch=%d",
            args.flat_div, len(flat_dataset), args.batch_size, steps_per_epoch,
        )
    else:
        # NPO mode: forget-only or forget+retain
        forget_dataset = ForgetDataset(args.forget_data, tokenizer, max_length=args.max_length)
        forget_loader = DataLoader(
            forget_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate, drop_last=True, num_workers=0,
        )

        if args.retain_data:
            retain_dataset = ForgetDataset(args.retain_data, tokenizer, max_length=args.max_length)
            retain_loader = DataLoader(
                retain_dataset, batch_size=args.batch_size, shuffle=True,
                collate_fn=collate, drop_last=True, num_workers=0,
            )
            steps_per_epoch = max(len(forget_loader), len(retain_loader))
            logger.info(
                "Forget: %d samples (%d steps), Retain: %d samples (%d steps), steps_per_epoch=%d",
                len(forget_dataset), len(forget_loader),
                len(retain_dataset), len(retain_loader), steps_per_epoch,
            )
        else:
            steps_per_epoch = len(forget_loader)
            logger.info(
                "Forget-only: %d samples, batch_size=%d, steps_per_epoch=%d",
                len(forget_dataset), args.batch_size, steps_per_epoch,
            )

    total_epochs = math.ceil(args.max_steps / steps_per_epoch)
    logger.info("total_epochs=%d", total_epochs)

    # -----------------------------------------------------------------------
    # Optimizer & Scheduler
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    if args.loss_type == "flat":
        mode_str = f"FLAT-{args.flat_div}"
    elif retain_loader:
        mode_str = "forget+retain"
    else:
        mode_str = "forget-only"

    logger.info(
        "Starting training [%s] (lr=%.1e, max_steps=%d)",
        mode_str, args.lr, args.max_steps,
    )
    model.train()
    device = next(model.parameters()).device
    global_step = 0
    optimizer.zero_grad()

    pbar = tqdm(total=args.max_steps, desc=mode_str, unit="step")

    if args.loss_type == "flat":
        # --- FLAT mode: forget data only, no retain, no ref model ---
        for epoch in range(total_epochs):
            for batch in flat_loader:
                forget_b = {k: v.to(device) for k, v in batch["forget"].items()}
                template_b = {k: v.to(device) for k, v in batch["template"].items()}

                loss, p_f, p_t = compute_flat_loss(model, forget_b, template_b, args.flat_div)
                (loss / args.grad_accum).backward()

                if (global_step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                global_step += 1
                lr_now = scheduler.get_last_lr()[0]
                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    forget=f"{-p_f.item():.4f}",
                    template=f"{-p_t.item():.4f}",
                    lr=f"{lr_now:.2e}",
                    epoch=epoch + 1,
                )

                if global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    logger.info("Saved checkpoint: %s (loss=%.4f, forget_prob=%.4f, template_prob=%.4f)",
                                ckpt_dir, loss.item(), -p_f.item(), -p_t.item())

                if global_step >= args.max_steps:
                    break
            if global_step >= args.max_steps:
                break

    elif retain_loader:
        # --- Forget + Retain mode ---
        for epoch in range(total_epochs):
            # Cycle the smaller loader so both run for steps_per_epoch
            if len(forget_loader) >= len(retain_loader):
                pair_iter = zip(forget_loader, itertools.cycle(retain_loader))
            else:
                pair_iter = zip(itertools.cycle(forget_loader), retain_loader)

            for forget_batch, retain_batch in pair_iter:
                forget_batch = {k: v.to(device) for k, v in forget_batch.items()}
                retain_batch = {k: v.to(device) for k, v in retain_batch.items()}

                loss_f = compute_npo_loss(model, forget_batch, beta=args.beta)
                loss_r = compute_retain_loss(model, retain_batch)
                loss = args.alpha_forget * loss_f + args.alpha_retain * loss_r
                (loss / args.grad_accum).backward()

                if (global_step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                global_step += 1
                lr_now = scheduler.get_last_lr()[0]
                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    loss_f=f"{loss_f.item():.4f}",
                    loss_r=f"{loss_r.item():.4f}",
                    lr=f"{lr_now:.2e}",
                    epoch=epoch + 1,
                )

                if global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    logger.info("Saved checkpoint: %s (loss=%.4f, loss_f=%.4f, loss_r=%.4f)",
                                ckpt_dir, loss.item(), loss_f.item(), loss_r.item())

                if global_step >= args.max_steps:
                    break
            if global_step >= args.max_steps:
                break
    else:
        # --- Forget-only mode (original behavior) ---
        for epoch in range(total_epochs):
            for batch in forget_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                loss = compute_npo_loss(model, batch, beta=args.beta)
                (loss / args.grad_accum).backward()

                if (global_step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                global_step += 1
                lr_now = scheduler.get_last_lr()[0]
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_now:.2e}", epoch=epoch + 1)

                if global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    logger.info("Saved checkpoint: %s", ckpt_dir)

                if global_step >= args.max_steps:
                    break
            if global_step >= args.max_steps:
                break

    pbar.close()

    # Save final checkpoint if not already saved
    if global_step % args.save_steps != 0:
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        logger.info("Saved final checkpoint: %s", ckpt_dir)

    logger.info("Training complete. Total steps: %d", global_step)


if __name__ == "__main__":
    main()
