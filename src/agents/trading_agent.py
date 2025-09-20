from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from ..utils.feature_engineering import build_prompt
from .rag_retriever import NewsRetriever
from .llm_interface import BaseLLM, MockLLM, FineTunedStubModel

@dataclass
class AgentConfig:
    # 可选策略
    # "zero_shot" | "few_shot" | "rag" | "rag_uncertainty"
    strategy: str = "rag"

    # 仅在 rag_uncertainty 生效（用于不确定性采样）
    n_samples: int = 5
    temperature: float = 0.6

    # RAG 检索条数
    rag_k: int = 3

    # few-shot 示例（仅在 strategy=="few_shot" 时注入）
    few_shots: List[str] = field(default_factory=lambda: [
        'Input: RSI_14 low, positive earnings surprise. Output: {"direction":"bullish","confidence":0.7,"reason":"Oversold + beat."}',
        'Input: RSI_14 high, product recall. Output: {"direction":"bearish","confidence":0.7,"reason":"Overbought + recall."}',
    ])

class TradingAgent:
    def __init__(self, llm: BaseLLM = None, cfg: AgentConfig = AgentConfig()) -> None:
        self.cfg = cfg
        self.llm = llm or MockLLM()
        self.retriever: NewsRetriever = NewsRetriever()

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        仅用训练集新闻拟合检索器，避免在推理时检索到未来信息（防泄露）。
        """
        docs = list((train_df.get("news") or pd.Series([])).astype(str).values)
        dates = list((train_df.get("date") or pd.Series([])).astype(str).values)
        self.retriever.fit(docs, dates)

    def _context_for_row(self, row: pd.Series) -> List[str]:
        """
        基于 ticker + 当日新闻 做简单查询，再由 retriever 内部完成相似检索。
        """
        query = f"{row.get('ticker','')} {row.get('news','')}"
        return self.retriever.get_context(query, k=self.cfg.rag_k)

    def _sample_decisions(self, prompt: str, n_samples: int) -> List[Dict[str, Any]]:
        return self.llm.generate(prompt, n_samples=n_samples, temperature=self.cfg.temperature)

    def _aggregate_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        置信度加权多数投票；均值置信度作为不确定性代理。
        """
        if not samples:
            return {"direction": "neutral", "confidence": 0.5, "reason": "no-samples"}
        dirs = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
        confs = []
        for s in samples:
            d = s.get("direction", "neutral")
            c = float(s.get("confidence", 0.5))
            if d in dirs:
                dirs[d] += c
            confs.append(c)
        direction = max(dirs.items(), key=lambda kv: kv[1])[0]
        confidence = float(np.mean(confs)) if confs else 0.5
        return {"direction": direction, "confidence": confidence, "reason": f"vote:{dirs} mean_conf:{confidence:.2f}"}

    def predict_day(self, day_df: pd.DataFrame) -> pd.DataFrame:
        """
        日内逐标的生成方向与置信度。
        - rag / rag_uncertainty: 注入 RAG 上下文
        - few_shot: 注入 few-shot 例子
        - zero_shot: 仅快照 + 当日新闻
        """
        rows = []
        use_rag = self.cfg.strategy in ["rag", "rag_uncertainty"]
        use_few_shot = (self.cfg.strategy == "few_shot")

        for _, row in day_df.iterrows():
            ctx = self._context_for_row(row) if use_rag else None

            prompt = build_prompt(
                row,
                few_shot_examples=(self.cfg.few_shots if use_few_shot else None),
                context_docs=ctx,
                use_rag=use_rag,  # 关键：训练/zero/few_shot 情况下为 False
            )

            if self.cfg.strategy == "rag_uncertainty":
                # 多样本采样以度量不确定性
                samples = self._sample_decisions(prompt, n_samples=max(1, self.cfg.n_samples))
                out = self._aggregate_samples(samples)
            else:
                # 其他策略只取单样本（更快）
                out = self._sample_decisions(prompt, n_samples=1)[0]

            rows.append({
                "date": row.get("date"),
                "ticker": row.get("ticker"),
                "direction": out["direction"],
                "confidence": out["confidence"],
                "reason": out.get("reason", ""),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def direction_to_signal(direction: str) -> int:
        if direction == "bullish":
            return 1
        if direction == "bearish":
            return -1
        return 0
