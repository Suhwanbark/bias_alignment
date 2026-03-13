"""Quick debug: check if float32 logits fix makes neg_log_ratio non-zero."""
import torch, os
from new_npo.src.model_utils import load_base_model, load_tokenizer, apply_lora, load_config
from new_npo.src.losses import get_batch_loss, npo_loss
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
forget_loader, _, _ = create_dataloaders(config, tokenizer)
batch = next(iter(forget_loader))
batch = {k: v.to(device) for k, v in batch.items()}

model.train()
outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
current_loss = get_batch_loss(outputs.logits, batch["labels"])

with torch.no_grad(), model.disable_adapter():
    ref_outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    ref_loss = get_batch_loss(ref_outputs.logits, batch["labels"])

neg = current_loss - ref_loss  # both should be float32 now
print(f"current_loss dtype: {current_loss.dtype}")
print(f"current_loss: {current_loss.tolist()}")
print(f"ref_loss:     {ref_loss.tolist()}")
print(f"neg_log_ratios: {neg.tolist()}")
print(f"Any non-zero: {(neg.abs() > 0).any().item()}")

loss = npo_loss(model, batch, beta=0.1)
print(f"NPO loss: {loss.item():.6f}")
