"""
Loss functions for BiasUnlearn NPO training.

Provides:
  - get_batch_loss: per-sequence cross-entropy loss
  - npo_loss: NPO forget loss (reduce biased response probability)
  - ce_retention_loss: standard CE retain loss (preserve balanced judgments)
  - kl_divergence_loss: forward KL to preserve general financial knowledge
  - combined_loss: weighted sum of all losses
"""

import torch
import torch.nn.functional as F


def get_batch_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Per-sequence cross-entropy loss (sum over valid tokens per sequence).

    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len) with -100 for masked positions

    Returns:
        Tensor of shape (batch_size,) with per-sequence losses.
    """
    # Shift for next-token prediction (float32 for NPO precision)
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()

    # Per-token CE loss (no reduction)
    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    loss_per_token = loss_per_token.view(shift_labels.size())

    # Mean over valid tokens per sequence (matching BiasUnlearn original)
    valid_mask = shift_labels != -100
    valid_count = valid_mask.sum(dim=1).clamp(min=1)
    loss_per_seq = (loss_per_token * valid_mask).sum(dim=1) / valid_count

    return loss_per_seq


def npo_loss(model, batch: dict, beta: float = 0.1) -> torch.Tensor:
    """
    NPO forget loss using disable_adapter() for reference model.

    L = -2/beta * mean(log_sigmoid(beta * (current_loss - ref_loss)))

    Reduces probability of biased responses while avoiding model collapse
    via sigmoid dampening.
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
    # Cast to float32 to avoid bf16 precision loss (diff rounds to 0)
    neg_log_ratios = current_loss.float() - ref_loss.float()
    loss = -F.logsigmoid(beta * neg_log_ratios).mean() * (2.0 / beta)

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

    Args:
        model: PeftModel with LoRA adapter
        kl_batch: dict with input_ids, attention_mask (no labels needed)

    Returns:
        Scalar KL divergence loss.
    """
    input_ids = kl_batch["input_ids"]
    attention_mask = kl_batch["attention_mask"]

    # Reference model (base without LoRA) forward
    with torch.no_grad(), model.disable_adapter():
        ref_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        ref_probs = F.softmax(ref_outputs.logits, dim=-1)

    # Current model (with LoRA) forward
    curr_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    curr_log_probs = F.log_softmax(curr_outputs.logits, dim=-1)

    # Forward KL: -sum(P_ref * log(P_theta))
    # Only compute on non-padding tokens
    token_kl = -(ref_probs * curr_log_probs).sum(dim=-1)  # (batch, seq_len)

    # Mask padding tokens
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

    # NPO+CE 2-loss mode: renormalize weights
    total = alpha_forget + alpha_retain
    return (alpha_forget / total) * l_forget + (alpha_retain / total) * l_retain
