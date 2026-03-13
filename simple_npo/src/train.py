"""
SimNPO training for Qwen3-30B-A3B-Instruct-2507.

Reference-free unlearning with length-normalized forget loss:
  L_total = alpha_f * L_forget(SimNPO) + alpha_r * L_retain(CE) + alpha_kl * L_KL

Key difference from BiasUnlearn NPO (new_npo):
  - Forget loss does NOT use reference model (no disable_adapter())
  - ~50% faster per forget step (single forward pass instead of two)
  - Better gradient weighting for hard vs easy forget samples
  - Configurable reward margin gamma (default 0)

Reference:
  Fan et al., "Simplicity Prevails: Rethinking Negative Preference
  Optimization for LLM Unlearning" (NeurIPS 2025, arXiv:2410.07163)

Usage:
    python -m simple_npo.src.train --config simple_npo/configs/qwen3_30b.yaml
    python -m simple_npo.src.train --config simple_npo/configs/qwen3_30b.yaml --max-steps 10
"""

import argparse
import itertools
import json
import logging
import os
import random
import re

import numpy as np
import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .model_utils import load_config, load_base_model, load_tokenizer, apply_lora
from .losses import sim_npo_loss, ce_retention_loss, kl_divergence_loss, combined_loss
from .dataset import create_dataloaders

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_json_from_text(text: str) -> dict | None:
    """Extract JSON object from model output text."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_bias_mini(model, tokenizer, eval_path: str) -> dict[str, float]:
    """
    Quick bias evaluation on eval_mini.jsonl (~19 samples).

    Generates model responses for each eval prompt and computes per-ticker
    bias scores. Used for dynamic swapping and early stopping decisions.
    """
    samples = []
    with open(eval_path) as f:
        for line in f:
            samples.append(json.loads(line))

    if not samples:
        return {}

    model.eval()
    device = next(model.parameters()).device
    ticker_decisions = {}

    with torch.no_grad():
        for sample in samples:
            messages = [{"role": "user", "content": sample["context"]}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)

            output_ids = model.generate(
                input_ids,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = output_ids[0][input_ids.shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)

            parsed = parse_json_from_text(response)
            if parsed:
                decision = parsed.get("decision", "").lower().strip()
                if decision in ("buy", "sell"):
                    ticker = sample["ticker"]
                    if ticker not in ticker_decisions:
                        ticker_decisions[ticker] = {"buy": 0, "sell": 0}
                    ticker_decisions[ticker][decision] += 1

    model.train()

    scores = {}
    for ticker, counts in ticker_decisions.items():
        total = counts["buy"] + counts["sell"]
        if total > 0:
            scores[ticker] = ((counts["buy"] - counts["sell"]) / total) * 100

    return scores


def load_original_scores(profile_report_path: str) -> dict[str, float]:
    """Load original bias scores from profile_report.json."""
    with open(profile_report_path) as f:
        report = json.load(f)
    return {t: info["bias_score"] for t, info in report["ticker_scores"].items()}


def check_swap_condition(
    eval_scores: dict[str, float],
    original_scores: dict[str, float],
    high_bias_tickers: list[str],
) -> bool:
    """
    Check if bias has reversed (over-correction).

    Returns True if originally buy-biased tickers now show sell bias on average,
    or vice versa.
    """
    buy_biased = [t for t in high_bias_tickers if original_scores.get(t, 0) > 0]
    sell_biased = [t for t in high_bias_tickers if original_scores.get(t, 0) < 0]

    if buy_biased:
        buy_scores = [eval_scores.get(t, 0) for t in buy_biased if t in eval_scores]
        if buy_scores and np.mean(buy_scores) < -10:
            return True

    if sell_biased:
        sell_scores = [eval_scores.get(t, 0) for t in sell_biased if t in eval_scores]
        if sell_scores and np.mean(sell_scores) > 10:
            return True

    return False


def early_stop_condition(
    eval_scores: dict[str, float],
    high_bias_tickers: list[str],
    threshold: float = 20.0,
) -> bool:
    """
    Check if mean |bias_score| of high-bias tickers is below threshold.

    Returns True when debiasing is sufficient.
    """
    abs_scores = [abs(eval_scores.get(t, 100)) for t in high_bias_tickers if t in eval_scores]
    if not abs_scores:
        return False
    return np.mean(abs_scores) < threshold


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SimNPO training (Qwen3-30B)")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps from config")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_steps is not None:
        config.max_steps = args.max_steps

    # Resolve data paths relative to simple_npo/ directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for attr in ("forget_data", "retain_data", "eval_mini_data", "profile_report"):
        path = getattr(config, attr)
        if path and not os.path.isabs(path):
            setattr(config, attr, os.path.join(base_dir, path))
    if config.kl_data and not os.path.isabs(config.kl_data):
        config.kl_data = os.path.join(base_dir, config.kl_data)
    if not os.path.isabs(config.output_dir):
        config.output_dir = os.path.join(base_dir, config.output_dir)

    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    # Save training config
    config_save_path = os.path.join(config.output_dir, "train_config.json")
    with open(config_save_path, "w") as f:
        json.dump(vars(config), f, indent=2)
    logger.info("Config saved to %s", config_save_path)

    # -----------------------------------------------------------------------
    # Load model & tokenizer
    # -----------------------------------------------------------------------
    tokenizer = load_tokenizer(config.model_name)
    model = load_base_model(config.model_name)
    model = apply_lora(model, config)

    # -----------------------------------------------------------------------
    # Create DataLoaders
    # -----------------------------------------------------------------------
    forget_loader, retain_loader, kl_loader = create_dataloaders(config, tokenizer)

    retain_mode = retain_loader is not None
    kl_mode = kl_loader is not None
    if retain_mode and kl_mode:
        loss_mode = "SimNPO+CE+KL"
    elif retain_mode:
        loss_mode = "SimNPO+CE"
    else:
        loss_mode = "SimNPO (forget-only)"
    logger.info(
        "Loss mode: %s (alpha_f=%.2f, alpha_r=%.2f, alpha_kl=%.2f, gamma=%.2f)",
        loss_mode, config.alpha_forget, config.alpha_retain, config.alpha_kl,
        config.gamma,
    )

    # -----------------------------------------------------------------------
    # Load evaluation data for dynamic swapping
    # -----------------------------------------------------------------------
    original_scores = {}
    high_bias_tickers = []
    if os.path.exists(config.profile_report):
        original_scores = load_original_scores(config.profile_report)
        high_bias_tickers = [t for t, s in original_scores.items() if abs(s) > 80]
        logger.info("Loaded %d high-bias tickers for swap/early-stop monitoring", len(high_bias_tickers))

    swapped = False

    # -----------------------------------------------------------------------
    # Optimizer & Scheduler
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    logger.info(
        "Starting training [%s] (beta=%.2f, gamma=%.2f, lr=%.1e, max_steps=%d, "
        "forget_bs=%d, retain_bs=%d)",
        loss_mode, config.beta, config.gamma, config.lr, config.max_steps,
        config.forget_batch_size, config.retain_batch_size,
    )
    model.train()
    device = next(model.parameters()).device
    global_step = 0
    optimizer.zero_grad()

    # Infinite iterators over dataloaders
    forget_iter = itertools.cycle(forget_loader)
    retain_iter = itertools.cycle(retain_loader) if retain_loader else None
    kl_iter = itertools.cycle(kl_loader) if kl_loader else None

    pbar = tqdm(total=config.max_steps, desc=f"SimNPO {loss_mode}", unit="step")

    while global_step < config.max_steps:
        # Sample batches
        forget_batch = next(forget_iter)
        forget_batch = {k: v.to(device) for k, v in forget_batch.items()}

        retain_batch = None
        if retain_iter:
            retain_batch = next(retain_iter)
            retain_batch = {k: v.to(device) for k, v in retain_batch.items()}

        kl_batch = None
        if kl_iter:
            kl_batch = next(kl_iter)
            kl_batch = {k: v.to(device) for k, v in kl_batch.items()}

        # Compute losses
        loss_f = sim_npo_loss(model, forget_batch, beta=config.beta, gamma=config.gamma)
        loss_r = ce_retention_loss(model, retain_batch) if retain_batch is not None else None
        loss_kl = kl_divergence_loss(model, kl_batch) if kl_batch is not None else None

        if loss_r is not None:
            loss = combined_loss(
                loss_f, loss_r, loss_kl,
                config.alpha_forget, config.alpha_retain, config.alpha_kl,
            )
        else:
            loss = loss_f

        # Backward
        (loss / config.grad_accum).backward()

        if (global_step + 1) % config.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1
        lr_now = scheduler.get_last_lr()[0]

        # Progress bar
        postfix = {
            "loss": f"{loss.item():.4f}",
            "L_f": f"{loss_f.item():.4f}",
            "lr": f"{lr_now:.2e}",
        }
        if loss_r is not None:
            postfix["L_r"] = f"{loss_r.item():.4f}"
        if loss_kl is not None:
            postfix["L_kl"] = f"{loss_kl.item():.4f}"
        pbar.update(1)
        pbar.set_postfix(postfix)

        # Periodic evaluation & checkpoint
        if global_step % config.eval_steps == 0:
            # Save checkpoint
            ckpt_dir = os.path.join(config.output_dir, f"checkpoint-{global_step}")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info(
                "Saved checkpoint: %s (loss=%.4f, L_f=%.4f)",
                ckpt_dir, loss.item(), loss_f.item(),
            )

            # NOTE: eval_mini inference disabled for speed.
            # Use saved checkpoints for post-hoc evaluation.

    pbar.close()

    # Save final checkpoint if not already saved
    if global_step % config.eval_steps != 0:
        ckpt_dir = os.path.join(config.output_dir, f"checkpoint-{global_step}")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        logger.info("Saved final checkpoint: %s", ckpt_dir)

    logger.info("Training complete. Total steps: %d", global_step)


if __name__ == "__main__":
    main()
