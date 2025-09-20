import json, random, re
from typing import List, Dict, Any, Optional

class BaseLLM:
    def generate(self, prompt: str, n_samples: int = 1, temperature: float = 0.7) -> List[Dict[str, Any]]:
        raise NotImplementedError

class MockLLM(BaseLLM):
    """
    A simple rule-based model so the pipeline can run offline.
    Heuristic sentiment from keywords; randomized confidence for diversity.
    """
    POS = ["beat","beats","upgrade","surge","record","growth","buyback","profit","raises guidance","strong"]
    NEG = ["miss","downgrade","lawsuit","probe","fraud","scandal","cuts guidance","weak","recall","layoff"]

    def generate(self, prompt: str, n_samples: int = 1, temperature: float = 0.7) -> List[Dict[str, Any]]:
        out = []
        news_block = self._extract_news(prompt).lower()
        pos_hits = sum(1 for w in self.POS if w in news_block)
        neg_hits = sum(1 for w in self.NEG if w in news_block)
        for _ in range(n_samples):
            rnd = random.random()
            if pos_hits > neg_hits:
                direction = "bullish"
                base_conf = 0.55 + 0.1 * min(3, pos_hits-neg_hits)
            elif neg_hits > pos_hits:
                direction = "bearish"
                base_conf = 0.55 + 0.1 * min(3, neg_hits-pos_hits)
            else:
                direction = "neutral"
                base_conf = 0.5
            # add randomness for 'uncertainty sampling'
            conf = min(0.99, max(0.01, base_conf + (rnd-0.5)*0.2*temperature))
            reason = f"Heuristic signal: pos={pos_hits}, neg={neg_hits}."
            out.append({"direction": direction, "confidence": conf, "reason": reason})
        return out

    @staticmethod
    def _extract_news(prompt: str) -> str:
        m = re.search(r"Today's News \(raw, may be empty\):\s+\"\"\"(.*?)\"\"\"", prompt, flags=re.S)
        return m.group(1) if m else ""

class FineTunedStubModel(BaseLLM):
    """
    Demonstrates a 'fine-tuned' behavior by biasing towards patterns learned from synthetic few-shot cues.
    Replace with actual PEFT/SFT/RLHF integration.
    """
    def __init__(self, bias: Optional[str] = None):
        self.bias = bias or "none"

    def generate(self, prompt: str, n_samples: int = 1, temperature: float = 0.7) -> List[Dict[str, Any]]:
        base = MockLLM().generate(prompt, n_samples, temperature)
        # simple biasing: if RSI<30 -> bullish tilt; if RSI>70 -> bearish tilt
        rsi_match = re.search(r"RSI_14: ([0-9\.]+)", prompt)
        rsi = float(rsi_match.group(1)) if rsi_match else 50.0
        for b in base:
            if rsi < 30 and b["direction"] != "bullish" and self.bias != "bearish_only":
                b["direction"] = "bullish"
                b["confidence"] = max(b["confidence"], 0.6)
                b["reason"] += " RSI<30 tilt."
            if rsi > 70 and b["direction"] != "bearish" and self.bias != "bullish_only":
                b["direction"] = "bearish"
                b["confidence"] = max(b["confidence"], 0.6)
                b["reason"] += " RSI>70 tilt."
        return base

# Placeholders for real models:
class OpenAIModel(BaseLLM):
    def __init__(self, model_name: str = "gpt-4o-mini", api_key_env: str = "OPENAI_API_KEY"):
        self.model_name = model_name
        self.api_key_env = api_key_env
    def generate(self, prompt: str, n_samples: int = 1, temperature: float = 0.7) -> List[Dict[str, Any]]:
        raise NotImplementedError("Implement OpenAI API call here.")

class HFModel(BaseLLM):
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.model_name = model_name
    def generate(self, prompt: str, n_samples: int = 1, temperature: float = 0.7) -> List[Dict[str, Any]]:
        raise NotImplementedError("Implement Transformers inference here.")
