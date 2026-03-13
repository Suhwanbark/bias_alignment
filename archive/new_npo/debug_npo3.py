"""Debug: simulate a few training steps and check neg_log_ratio evolution."""
import torch, os
from new_npo.src.model_utils import load_base_model, load_tokenizer, apply_lora, load_config
from new_npo.src.losses import get_batch_loss, npo_loss, ce_retention_loss, combined_loss
from new_npo.src.dataset import create_dataloaders

config = load_config("new_npo/configs/qwen3_30b.yaml")
base_dir = "/data/llm-bias-in-finance/new_npo"
for attr in ("forget_data", "retain_data", "eval_mini_data", "profile_report"):
    path = getattr(config, attr)
    if path and not os.path.isabs(path):
        setattr(config, attr, os.path.join(base_dir, path))

tokenizer = load_tokenizer(config.model_name)
model = load_base_model(config.model_name)
model = apply_lora(model, config)
device = next(model.parameters()).device

forget_loader, retain_loader, _ = create_dataloaders(config, tokenizer)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=5e-5, weight_decay=0.01,
)

forget_iter = iter(forget_loader)
retain_iter = iter(retain_loader)

model.train()
for step in range(10):
    try:
        f_batch = next(forget_iter)
    except StopIteration:
        forget_iter = iter(forget_loader)
        f_batch = next(forget_iter)
    try:
        r_batch = next(retain_iter)
    except StopIteration:
        retain_iter = iter(retain_loader)
        r_batch = next(retain_iter)

    f_batch = {k: v.to(device) for k, v in f_batch.items()}
    r_batch = {k: v.to(device) for k, v in r_batch.items()}

    # Check neg_log_ratio before step
    with torch.no_grad():
        outputs = model(input_ids=f_batch["input_ids"], attention_mask=f_batch["attention_mask"])
        cur = get_batch_loss(outputs.logits, f_batch["labels"])
        with model.disable_adapter():
            ref_out = model(input_ids=f_batch["input_ids"], attention_mask=f_batch["attention_mask"])
            ref = get_batch_loss(ref_out.logits, f_batch["labels"])
        neg = cur - ref

    # Compute losses and step
    optimizer.zero_grad()
    loss_f = npo_loss(model, f_batch, beta=10.0)
    loss_r = ce_retention_loss(model, r_batch)
    loss = combined_loss(loss_f, loss_r, None, 0.4, 0.4, 0.2)
    loss.backward()
    optimizer.step()

    print(f"Step {step}: L_f={loss_f.item():.4f}  L_r={loss_r.item():.4f}  "
          f"neg_log_ratio={neg.tolist()}  "
          f"max|neg|={neg.abs().max().item():.8f}")
