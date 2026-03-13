"""
Model utilities for Qwen3-30B NPO training.

Handles config loading, model/tokenizer initialization, and LoRA setup.
"""

from dataclasses import dataclass, field
from typing import Optional

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel


@dataclass
class TrainConfig:
    # Model
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    max_length: int = 512

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    # Training
    lr: float = 2e-5
    beta: float = 0.1
    max_steps: int = 1000
    save_steps: int = 50
    eval_steps: int = 50
    warmup_steps: int = 10
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    seed: int = 42

    # Loss weights
    alpha_forget: float = 0.4
    alpha_retain: float = 0.4
    alpha_kl: float = 0.2

    # Batch sizes
    forget_batch_size: int = 4
    retain_batch_size: int = 28
    kl_batch_size: int = 4

    # Data paths
    forget_data: str = ""
    retain_data: str = ""
    kl_data: Optional[str] = None
    eval_mini_data: str = ""
    profile_report: str = ""

    # Output
    output_dir: str = "results/qwen3-30b-a3b-instruct-2507"
    tensor_parallel: int = 2


def load_config(yaml_path: str) -> TrainConfig:
    """Load training config from YAML file."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    t = raw["training"]
    return TrainConfig(
        model_name=raw["model"]["name"],
        max_length=int(raw["model"]["max_length"]),
        lora_r=int(raw["lora"]["r"]),
        lora_alpha=int(raw["lora"]["alpha"]),
        lora_dropout=float(raw["lora"]["dropout"]),
        target_modules=raw["lora"]["target_modules"],
        lr=float(t["lr"]),
        beta=float(t["beta"]),
        max_steps=int(t["max_steps"]),
        save_steps=int(t["save_steps"]),
        eval_steps=int(t["eval_steps"]),
        warmup_steps=int(t["warmup_steps"]),
        grad_accum=int(t["grad_accum"]),
        max_grad_norm=float(t["max_grad_norm"]),
        weight_decay=float(t["weight_decay"]),
        seed=int(t["seed"]),
        alpha_forget=float(raw["loss_weights"]["alpha_forget"]),
        alpha_retain=float(raw["loss_weights"]["alpha_retain"]),
        alpha_kl=float(raw["loss_weights"]["alpha_kl"]),
        forget_batch_size=int(raw["batch_sizes"]["forget"]),
        retain_batch_size=int(raw["batch_sizes"]["retain"]),
        kl_batch_size=int(raw["batch_sizes"]["kl"]),
        forget_data=raw["data"]["forget"],
        retain_data=raw["data"]["retain"],
        kl_data=raw["data"].get("kl"),
        eval_mini_data=raw["data"]["eval_mini"],
        profile_report=raw["data"]["profile_report"],
        output_dir=raw["output_dir"],
        tensor_parallel=int(raw.get("tensor_parallel", 2)),
    )


def load_base_model(model_name: str) -> AutoModelForCausalLM:
    """Load base model with bf16, SDPA, and auto device mapping."""
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load tokenizer and set pad_token if missing."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def apply_lora(model: AutoModelForCausalLM, config: TrainConfig) -> PeftModel:
    """Apply LoRA adapter to the model."""
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
