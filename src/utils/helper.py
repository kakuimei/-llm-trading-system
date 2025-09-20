# src/run.py
import argparse, os, json, random, shutil, re
import numpy as np
import pandas as pd
import torch
from subprocess import run as sh
from .data_loader import make_training_samples

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_samples(df, path):
    samples = make_training_samples(df)  # 内部需确保 use_rag=False
    pd.DataFrame(samples).to_json(path, orient="records", lines=True, force_ascii=False)

def train_sft_if_needed(train_file, valid_file, model_name, out_dir, epochs=3, lr=2e-5, bf16=True):
    if os.path.isdir(out_dir) and os.path.exists(os.path.join(out_dir, "config.json")):
        print(f"[SFT] Found existing checkpoint: {out_dir}, skip training.")
        return
    cmd = [
        "python", "sft_train.py",
        "--model_name", model_name,
        "--train_file", train_file,
        "--valid_file", valid_file,
        "--output_dir", out_dir,
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch", "1", "--grad_accum", "16"
    ]
    if bf16: cmd.append("--bf16")
    print("[SFT] Running:", " ".join(cmd))
    sh(cmd, check=True)

def train_peft_if_needed(train_file, valid_file, model_name, out_dir, epochs=3, lr=1e-4, bf16=True, load_4bit=True):
    if os.path.isdir(out_dir) and os.path.exists(os.path.join(out_dir, "adapter_config.json")):
        print(f"[PEFT] Found existing checkpoint: {out_dir}, skip training.")
        return
    cmd = [
        "python", "peft_train_lora.py",
        "--model_name", model_name,
        "--train_file", train_file,
        "--valid_file", valid_file,
        "--output_dir", out_dir,
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch", "1", "--grad_accum", "16",
        "--target_modules", "q_proj,v_proj"
    ]
    if bf16: cmd.append("--bf16")
    if load_4bit: cmd.append("--load_4bit")
    print("[PEFT] Running:", " ".join(cmd))
    sh(cmd, check=True)

def rename_equity(src_path, dst_path):
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
