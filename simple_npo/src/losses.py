"""
Loss functions for NPO (Negative Preference Optimization) training.

Provides:
  - get_batch_loss: per-sequence cross-entropy loss (sum over valid tokens)
  - npo_loss: NPO forget loss with reference model (disable_adapter)
  - ce_retention_loss: standard CE retain loss
  - kl_divergence_loss: forward KL to preserve general knowledge
  - combined_loss: weighted sum
"""

import torch
import torch.nn.functional as F


def get_batch_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Per-sequence cross-entropy loss (mean over valid tokens per sequence).

    This already provides length-normalized negative log probability:
      get_batch_loss = -1/|y| * log π_θ(y|x)

    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len) with -100 for masked positions

    Returns:
        Tensor of shape (batch_size,) with per-sequence mean losses.
    """
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()

    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    loss_per_token = loss_per_token.view(shift_labels.size())

    valid_mask = shift_labels != -100
    valid_count = valid_mask.sum(dim=1).clamp(min=1)
    loss_per_seq = (loss_per_token * valid_mask).sum(dim=1) / valid_count

    return loss_per_seq


def sim_npo_loss(
    model,
    batch: dict,
    beta: float = 0.1,
    gamma: float = 0.0,
) -> torch.Tensor:
    """
    SimNPO forget loss — reference-free, length-normalized.

    L_SimNPO = -2/β * E[log σ(-β/|y| * log π_θ(y|x) - γ)]

    Since get_batch_loss already computes -1/|y| * log π_θ(y|x),
    this simplifies to:
        L = -2/β * mean(log σ(β * current_loss - γ))

    Key advantages over NPO:
      - No reference model forward pass (saves ~50% compute per forget step)
      - Better gradient weighting: hard-to-forget samples (high π_θ) get more weight
      - Length normalization prevents bias toward short sequences

    Args:
        model: PeftModel (LoRA adapter is active but ref model is NOT used)
        batch: dict with input_ids, labels, attention_mask
        beta: temperature parameter (controls unlearning aggressiveness)
        gamma: reward margin (default 0, higher = more aggressive unlearning)
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    current_loss = get_batch_loss(outputs.logits, labels)

    # SimNPO: no reference model, just use current loss with margin
    # current_loss = -1/|y| * log π_θ(y|x), so β * current_loss = -β/|y| * log π_θ
    loss = -F.logsigmoid(beta * current_loss.float() - gamma).mean() * (2.0 / beta)

    return loss


def ce_retention_loss(model, batch: dict) -> torch.Tensor:
    """Standard CE loss on completion tokens — preserves balanced judgments."""
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return outputs.loss


def kl_divergence_loss(model, kl_batch: dict) -> torch.Tensor:
    """
    Forward KL divergence: KL(P_ref || P_theta).

    Preserves the base model's token-level distribution on general financial text.
    Uses disable_adapter() to obtain reference model probabilities.

    Note: KL loss still uses the reference model (base without LoRA) since
    we need the original distribution as target. Only the forget loss is
    reference-free in SimNPO.
    """
    input_ids = kl_batch["input_ids"]
    attention_mask = kl_batch["attention_mask"]

    with torch.no_grad(), model.disable_adapter():
        ref_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        ref_probs = F.softmax(ref_outputs.logits, dim=-1)

    curr_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    curr_log_probs = F.log_softmax(curr_outputs.logits, dim=-1)

    token_kl = -(ref_probs * curr_log_probs).sum(dim=-1)

    mask = attention_mask.float()
    masked_kl = (token_kl * mask).sum() / mask.sum().clamp(min=1)

    return masked_kl


def combined_loss(
    l_forget: torch.Tensor,
    l_retain: torch.Tensor,
    l_kl: torch.Tensor | None,
    alpha_forget: float = 0.4,
    alpha_retain: float = 0.4,
    alpha_kl: float = 0.2,
) -> torch.Tensor:
    """
    Weighted combination of losses.

    If l_kl is None (no KL data), renormalizes forget+retain weights.
    """
    if l_kl is not None:
        return alpha_forget * l_forget + alpha_retain * l_retain + alpha_kl * l_kl

    total = alpha_forget + alpha_retain
    return (alpha_forget / total) * l_forget + (alpha_retain / total) * l_retain
