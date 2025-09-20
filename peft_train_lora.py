# peft_train_lora.py
import os, json, argparse, random
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_text(example, add_reason=False):
    prompt = example["input"]
    out = example["output"]
    ans = {"direction": out.get("direction", "neutral"),
           "confidence": float(out.get("confidence", 0.5))}
    if add_reason and "reason" in out:
        ans["reason"] = out["reason"]
    target = json.dumps(ans, ensure_ascii=False)
    return f"{prompt}\n\nAnswer:\n{target}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="e.g. meta-llama/Llama-3-8B-Instruct or mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--train_file", type=str, default="./outputs/datasets/train.jsonl")
    parser.add_argument("--valid_file", type=str, default="./outputs/datasets/valid.jsonl")
    parser.add_argument("--output_dir", type=str, default="./outputs/peft_lora_ckpt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj", 
                        help="comma-separated; common: q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--load_4bit", action="store_true",
                        help="Use bitsandbytes 4-bit for base model to save memory.")
    args = parser.parse_args()

    set_seed(args.seed)

    ds_train = load_dataset("json", data_files=args.train_file, split="train")
    ds_valid = load_dataset("json", data_files=args.valid_file, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(examples):
        texts = [build_text(ex, add_reason=False) for ex in examples]
        toks = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_len,
            return_tensors=None,
        )
        toks["labels"] = toks["input_ids"].copy()
        return toks

    ds_train = ds_train.map(preprocess, batched=True, remove_columns=ds_train.column_names)
    ds_valid = ds_valid.map(preprocess, batched=True, remove_columns=ds_valid.column_names)

    # 加载基础模型（可选4-bit量化以省显存）
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if args.bf16 else None,
            device_map="auto",
        )

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        bf16=args.bf16,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()