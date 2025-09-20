# src/run.py
import argparse, os, json, random, shutil, re
import numpy as np
import pandas as pd
import torch
from subprocess import run as sh
from .config import Config
from .utils.data_loader import load_data, time_split, make_training_samples
from .utils.backtest import run_backtest
from .agents.trading_agent import TradingAgent, AgentConfig
from .agents.llm_interface import BaseLLM, MockLLM
from peft_train_lora import main as train_peft_if_needed
from sft_train import main as train_sft_if_needed
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
        # 尽力从生成文本中抓取 JSON（容错）
        # 常见格式：... \n\nAnswer:\n{...json...}
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
        from transformers import StoppingCriteria, StoppingCriteriaList
        stop_seq = None  # 可按需加 stopwords
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
        # 取每条对应的新生成部分（这里简单做；已经足够鲁棒）
        res = []
        for t in texts:
            # 仅截取 Answer: 之后
            if "Answer:" in t:
                t = t.split("Answer:", 1)[1]
            res.append(self._post(t))
        return res
# ------------------------------ Main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--base_model', default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument('--do_train', action='store_true', help="若提供则同时跑 SFT 与 PEFT 训练")
    ap.add_argument('--seed', type=int, default=42)
    # 回测与策略
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

    # 2) 导出训练/验证样本（仅用于训练与早停；推理阶段由 Agent+RAG 处理）
    ensure_dir("./outputs/datasets")
    train_jsonl = "./outputs/datasets/train.jsonl"
    valid_jsonl = "./outputs/datasets/valid.jsonl"
    save_samples(train, train_jsonl)
    save_samples(valid, valid_jsonl)

    # 3) 训练
    sft_ckpt = "./outputs/sft_ckpt"
    peft_ckpt = "./outputs/peft_lora_ckpt"
    if args.do_train:
        train_sft_if_needed(train_jsonl, valid_jsonl, args.base_model, sft_ckpt)
        train_peft_if_needed(train_jsonl, valid_jsonl, args.base_model, peft_ckpt)

    # 4) 构建三种 LLM：Mock（基线）/ SFT / PEFT
    runners = []
    # 4.1 Mock
    runners.append(("mock", MockLLM()))
    # 4.2 SFT（若存在）
    if os.path.isdir(sft_ckpt) and os.path.exists(os.path.join(sft_ckpt, "config.json")):
        runners.append(("sft", HFInferenceLLM(sft_ckpt)))
    # 4.3 PEFT（若存在）
    if os.path.isdir(peft_ckpt) and (os.path.exists(os.path.join(peft_ckpt, "adapter_config.json")) or
                                     os.path.exists(os.path.join(peft_ckpt, "config.json"))):
        # 说明：大多数 PEFT 保存的目录同样能用 AutoModelForCausalLM.from_pretrained 加载推理
        runners.append(("peft", HFInferenceLLM(peft_ckpt)))

    # 5) 对比回测（统一策略/超参）
    ensure_dir("./outputs")
    comparison = {}
    for name, llm in runners:
        print(f"[Backtest] Running strategy={args.strategy} with model={name}")
        agent = TradingAgent(
            llm=llm,
            cfg=AgentConfig(strategy=args.strategy, n_samples=args.n_samples, temperature=args.temperature)
        )
        agent.fit(train)  # RAG 仅用 train 拟合
        result_valid = run_backtest(valid, agent, trading_cost_bps=cfg.trading_cost_bps)
        result_test  = run_backtest(test,  agent, trading_cost_bps=cfg.trading_cost_bps)

        # 重命名权益曲线，避免相互覆盖
        rename_equity(result_valid["equity_path"], f"./outputs/equity_curve_valid_{name}.png")
        rename_equity(result_test["equity_path"],  f"./outputs/equity_curve_test_{name}.png")

        comparison[name] = {
            "valid": result_valid["stats"],
            "test":  result_test["stats"],
            "equity_valid_png": f"./outputs/equity_curve_valid_{name}.png",
            "equity_test_png":  f"./outputs/equity_curve_test_{name}.png",
        }

    # 6) 汇总结果
    with open("./outputs/results.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # 7) 控制台简表
    def fmt(s): return {k: round(v, 4) for k, v in s.items()}
    print("\n=== Comparison (VALID) ===")
    for k, v in comparison.items():
        print(k, fmt(v["valid"]))
    print("\n=== Comparison (TEST) ===")
    for k, v in comparison.items():
        print(k, fmt(v["test"]))
    print("\nSaved:",
          "./outputs/results.json, and equity curves per model under ./outputs/")

if __name__ == "__main__":
    main()
