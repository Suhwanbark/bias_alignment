"""Debug NPO loss gradient flow."""
import torch
import json
from new_npo.src.model_utils import load_base_model, load_tokenizer, apply_lora, load_config
from new_npo.src.losses import get_batch_loss, npo_loss, ce_retention_loss
from new_npo.src.dataset import create_dataloaders

config = load_config("new_npo/configs/qwen3_30b.yaml")

# Resolve paths (same logic as train.py)
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))  # new_npo/
for attr in ("forget_data", "retain_data", "eval_mini_data", "profile_report"):
    path = getattr(config, attr)
    if path and not os.path.isabs(path):
        setattr(config, attr, os.path.join(base_dir, path))

tokenizer = load_tokenizer(config.model_name)
model = load_base_model(config.model_name)
model = apply_lora(model, config)
device = next(model.parameters()).device

forget_loader, retain_loader, _ = create_dataloaders(config, tokenizer)

# Get one batch
forget_batch = next(iter(forget_loader))
forget_batch = {k: v.to(device) for k, v in forget_batch.items()}

retain_batch = next(iter(retain_loader))
retain_batch = {k: v.to(device) for k, v in retain_batch.items()}

print("=" * 60)
print("BATCH INFO")
print("=" * 60)
print(f"Forget input_ids shape: {forget_batch['input_ids'].shape}")
print(f"Forget labels shape: {forget_batch['labels'].shape}")
valid_tokens = (forget_batch['labels'] != -100).sum(dim=1)
print(f"Forget valid tokens per sample: {valid_tokens.tolist()}")
print(f"Retain input_ids shape: {retain_batch['input_ids'].shape}")

# 1. Check get_batch_loss output
print("\n" + "=" * 60)
print("GET_BATCH_LOSS (current model with LoRA)")
print("=" * 60)
model.train()
outputs = model(input_ids=forget_batch["input_ids"], attention_mask=forget_batch["attention_mask"])
current_loss = get_batch_loss(outputs.logits, forget_batch["labels"])
print(f"current_loss (per-sample): {current_loss.tolist()}")
print(f"current_loss mean: {current_loss.mean().item():.4f}")

# 2. Check reference model loss
print("\n" + "=" * 60)
print("GET_BATCH_LOSS (reference model without LoRA)")
print("=" * 60)
with torch.no_grad(), model.disable_adapter():
    ref_outputs = model(input_ids=forget_batch["input_ids"], attention_mask=forget_batch["attention_mask"])
    ref_loss = get_batch_loss(ref_outputs.logits, forget_batch["labels"])
print(f"ref_loss (per-sample): {ref_loss.tolist()}")
print(f"ref_loss mean: {ref_loss.mean().item():.4f}")

# 3. Check neg_log_ratios
print("\n" + "=" * 60)
print("NEG_LOG_RATIOS")
print("=" * 60)
neg_log_ratios = current_loss - ref_loss
print(f"neg_log_ratios: {neg_log_ratios.tolist()}")
print(f"beta * neg_log_ratios: {(0.1 * neg_log_ratios).tolist()}")
print(f"logsigmoid(beta * neg): {torch.nn.functional.logsigmoid(0.1 * neg_log_ratios).tolist()}")

# 4. Compute NPO loss and check gradient
print("\n" + "=" * 60)
print("NPO LOSS & GRADIENT")
print("=" * 60)
model.zero_grad()
loss_f = npo_loss(model, forget_batch, beta=0.1)
print(f"NPO loss: {loss_f.item():.4f}")
loss_f.backward()

# Check LoRA gradient norms
lora_grad_norms = {}
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        lora_grad_norms[name] = param.grad.norm().item()

print(f"\nLoRA layers with gradients: {len(lora_grad_norms)}")
total_grad_norm = sum(v**2 for v in lora_grad_norms.values()) ** 0.5
print(f"Total grad norm (NPO): {total_grad_norm:.6f}")
for name, norm in sorted(lora_grad_norms.items(), key=lambda x: -x[1])[:5]:
    print(f"  {name}: {norm:.6f}")

# 5. Compare with retain CE loss gradient
print("\n" + "=" * 60)
print("RETAIN CE LOSS & GRADIENT")
print("=" * 60)
model.zero_grad()
loss_r = ce_retention_loss(model, retain_batch)
print(f"Retain CE loss: {loss_r.item():.4f}")
loss_r.backward()

retain_grad_norms = {}
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        retain_grad_norms[name] = param.grad.norm().item()

total_retain_grad = sum(v**2 for v in retain_grad_norms.values()) ** 0.5
print(f"Total grad norm (Retain): {total_retain_grad:.6f}")

print("\n" + "=" * 60)
print("GRADIENT RATIO (NPO / Retain)")
print("=" * 60)
print(f"NPO grad norm:    {total_grad_norm:.6f}")
print(f"Retain grad norm:  {total_retain_grad:.6f}")
print(f"Ratio NPO/Retain: {total_grad_norm / total_retain_grad:.4f}")

# 6. Check combined loss gradient
print("\n" + "=" * 60)
print("COMBINED LOSS (0.5 * NPO + 0.5 * Retain)")
print("=" * 60)
model.zero_grad()
loss_f2 = npo_loss(model, forget_batch, beta=0.1)
loss_r2 = ce_retention_loss(model, retain_batch)
combined = 0.5 * loss_f2 + 0.5 * loss_r2
print(f"NPO:     {loss_f2.item():.4f}")
print(f"Retain:  {loss_r2.item():.4f}")
print(f"Combined: {combined.item():.4f}")
combined.backward()

combined_grad_norms = {}
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        combined_grad_norms[name] = param.grad.norm().item()
total_combined = sum(v**2 for v in combined_grad_norms.values()) ** 0.5
print(f"Total grad norm (Combined): {total_combined:.6f}")

# Check if NPO gradient direction opposes retain
print("\n" + "=" * 60)
print("GRADIENT DIRECTION ANALYSIS")
print("=" * 60)
# Recompute individual gradients
model.zero_grad()
loss_f3 = npo_loss(model, forget_batch, beta=0.1)
loss_f3.backward()
npo_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}

model.zero_grad()
loss_r3 = ce_retention_loss(model, retain_batch)
loss_r3.backward()

cosine_sims = []
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None and name in npo_grads:
        cos = torch.nn.functional.cosine_similarity(
            npo_grads[name].flatten().unsqueeze(0),
            param.grad.flatten().unsqueeze(0)
        ).item()
        cosine_sims.append((name, cos))

avg_cos = sum(c for _, c in cosine_sims) / len(cosine_sims) if cosine_sims else 0
print(f"Average cosine similarity (NPO vs Retain grads): {avg_cos:.4f}")
print(f"  > 0 = same direction, < 0 = opposing")
for name, cos in sorted(cosine_sims, key=lambda x: x[1])[:3]:
    print(f"  Most opposing: {name}: {cos:.4f}")
for name, cos in sorted(cosine_sims, key=lambda x: -x[1])[:3]:
    print(f"  Most aligned:  {name}: {cos:.4f}")
