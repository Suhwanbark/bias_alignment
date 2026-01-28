"""
DPO 훈련 스크립트 (간단 버전)

사용법:
    python train_dpo.py \
        --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --data data/dpo_nvidia.jsonl \
        --output models/nemotron-debiased
"""

import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig
import torch


def main():
    parser = argparse.ArgumentParser(description="DPO 훈련")
    parser.add_argument("--model", type=str, required=True, help="베이스 모델 ID 또는 경로")
    parser.add_argument("--data", type=str, required=True, help="DPO 데이터 JSONL 파일")
    parser.add_argument("--output", type=str, default="./output", help="출력 디렉토리")
    parser.add_argument("--epochs", type=int, default=1, help="훈련 에폭")
    parser.add_argument("--batch-size", type=int, default=1, help="배치 사이즈")
    parser.add_argument("--lr", type=float, default=5e-7, help="학습률")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--use-4bit", action="store_true", help="4bit 양자화 사용")
    args = parser.parse_args()

    print(f"\n{'─'*60}")
    print(f"DPO 훈련")
    print(f"{'─'*60}")
    print(f"모델: {args.model}")
    print(f"데이터: {args.data}")
    print(f"출력: {args.output}")
    print(f"{'─'*60}\n")

    # 1. 데이터 로드
    print("데이터 로드 중...")
    dataset = load_dataset("json", data_files=args.data, split="train")
    print(f"  샘플 수: {len(dataset)}")

    # 2. 모델 로드
    print("모델 로드 중...")

    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. LoRA 설정
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. DPO 설정
    training_args = DPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        beta=args.beta,
        logging_steps=1,
        save_steps=50,
        bf16=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        max_length=1024,
        max_prompt_length=512,
    )

    # 5. 훈련
    print("훈련 시작...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # 6. 저장
    print(f"\n모델 저장: {args.output}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output)

    print(f"\n{'─'*60}")
    print(f"훈련 완료!")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
