# src/run.py
import argparse, os, json, random, shutil, re, subprocess
import numpy as np
import pandas as pd
import torch
from .config import Config
from .utils.data_loader import load_data, time_split, make_training_samples
from .utils.backtest import run_backtest
from .agents.trading_agent import TradingAgent, AgentConfig
from .agents.llm_interface import BaseLLM, MockLLM
from .utils.helper import ensure_dir, rename_equity, save_samples

# -------- Minimal HF inference wrapper (loads SFT/PEFT checkpoints) --------
class HFInferenceLLM(BaseLLM):
    """
    轻量推理封装：从本地 checkpoint 载入 tokenizer+model，按我们训练时的格式生成：
    传入 prompt -> 自动补上 '\n\nAnswer:\n' 引导 -> 解析 JSON 输出
    """
    def __init__(self, model_path: str, device_map: str = "auto"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map)
        self.model.eval()

    def _post(self, text: str) -> dict:
        m = re.search(r"\{.*\}", text, flags=re.S)
        js = {"direction": "neutral", "confidence": 0.5, "reason": "parse-fallback"}
        if m:
            try:
                obj = json.loads(m.group(0))
                d = str(obj.get("direction", "neutral")).lower()
                c = float(obj.get("confidence", 0.5))
                r = obj.get("reason", "")
                if d not in ("bullish","bearish","neutral"):
                    d = "neutral"
                c = float(np.clip(c, 0.0, 1.0))
                js = {"direction": d, "confidence": c, "reason": r}
            except Exception:
                pass
        return js

    def generate(self, prompt: str, n_samples: int = 1, temperature: float = 0.7):
        prefix = prompt.rstrip() + "\n\nAnswer:\n"
        inputs = self.tok([prefix]*n_samples, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            outs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=float(temperature),
                top_p=0.95,
                eos_token_id=self.tok.eos_token_id,
            )
        texts = self.tok.batch_decode(outs, skip_special_tokens=True)
        res = []
        for t in texts:
            if "Answer:" in t:
                t = t.split("Answer:", 1)[1]
            res.append(self._post(t))
        return res

# --------- Training wrappers (call external scripts via subprocess) ---------
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
    if bf16:
        cmd.append("--bf16")
    print("[SFT] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

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
    if bf16:
        cmd.append("--bf16")
    if load_4bit:
        cmd.append("--load_4bit")
    print("[PEFT] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# ------------------------------ Main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--base_model', default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument('--do_train', action='store_true', help="若提供则同时跑 SFT 与 PEFT 训练")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--strategy', default='rag_uncertainty',
                    choices=['zero_shot','few_shot','rag','rag_uncertainty'])
    ap.add_argument('--n_samples', type=int, default=5)
    ap.add_argument('--temperature', type=float, default=0.6)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    cfg = Config()

    # 1) 数据与切分
    df = load_data(args.data, date_col=cfg.date_col, ticker_col=cfg.ticker_col)
    train, valid, test = time_split(df, start=args.start, end=args.end)

    # 2) 导出训练/验证样本
    ensure_dir("./outputs/datasets")
    train_jsonl = "./outputs/datasets/train.jsonl"
    valid_jsonl = "./outputs/datasets/valid.jsonl"
    print("Train df shape:", train.shape)
    print(train.head())
    save_samples(train, train_jsonl)
    save_samples(valid, valid_jsonl)

    # 3) 训练
    sft_ckpt = "./outputs/sft_ckpt"
    peft_ckpt = "./outputs/peft_lora_ckpt"
    if args.do_train:
        train_sft_if_needed(train_jsonl, valid_jsonl, args.base_model, sft_ckpt)
        train_peft_if_needed(train_jsonl, valid_jsonl, args.base_model, peft_ckpt)

    # 4) 构建三种 LLM：Mock / SFT / PEFT
    runners = []
    runners.append(("mock", MockLLM()))
    if os.path.isdir(sft_ckpt) and os.path.exists(os.path.join(sft_ckpt, "config.json")):
        runners.append(("sft", HFInferenceLLM(sft_ckpt)))
    if os.path.isdir(peft_ckpt) and (os.path.exists(os.path.join(peft_ckpt, "adapter_config.json"))
                                     or os.path.exists(os.path.join(peft_ckpt, "config.json"))):
        runners.append(("peft", HFInferenceLLM(peft_ckpt)))

    # 5) 回测对比
    ensure_dir("./outputs")
    comparison = {}
    for name, llm in runners:
        print(f"[Backtest] Running strategy={args.strategy} with model={name}")
        agent = TradingAgent(
            llm=llm,
            cfg=AgentConfig(strategy=args.strategy, n_samples=args.n_samples, temperature=args.temperature)
        )
        agent.fit(train)
        result_valid = run_backtest(valid, agent, trading_cost_bps=cfg.trading_cost_bps)
        result_test  = run_backtest(test, agent, trading_cost_bps=cfg.trading_cost_bps)

        rename_equity(result_valid["equity_path"], f"./outputs/equity_curve_valid_{name}.png")
        rename_equity(result_test["equity_path"], f"./outputs/equity_curve_test_{name}.png")

        comparison[name] = {
            "valid": result_valid["stats"],
            "test":  result_test["stats"],
            "equity_valid_png": f"./outputs/equity_curve_valid_{name}.png",
            "equity_test_png":  f"./outputs/equity_curve_test_{name}.png",
        }

    with open("./outputs/results.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    def fmt(s): return {k: round(v, 4) for k, v in s.items()}
    print("\n=== Comparison (VALID) ===")
    for k, v in comparison.items():
        print(k, fmt(v["valid"]))
    print("\n=== Comparison (TEST) ===")
    for k, v in comparison.items():
        print(k, fmt(v["test"]))
    print("\nSaved:", "./outputs/results.json, and equity curves per model under ./outputs/")

if __name__ == "__main__":
    main()
