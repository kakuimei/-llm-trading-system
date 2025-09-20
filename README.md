# LLM-Driven Daily Trading System

This project implements a **daily trading system** that integrates a Large Language Model (LLM) with market data, news, and fundamentals.
It follows the requested structure: data processing, LLM agent (prompt strategies & fine-tuning stubs), evaluation/backtest, risk management,
and optional advanced components (RAG retrieval + uncertainty-aware position sizing).

> Note: The default LLM is a `MockLLM` so that the pipeline runs without external services. You can replace it with a real OpenAI/Transformers model by implementing `OpenAIModel` or `HFModel` in `src/agents/llm_interface.py`.

## Project Layout
```
llm_trading_system/
  data/
    data.csv                        # Put your dataset here (date,ticker,open,high,low,close,volume,news,pe,roe,...)
  src/
    agents/
      trading_agent.py
      rag_retriever.py
      llm_interface.py
    utils/
      data_loader.py
      feature_engineering.py
      portfolio.py
      metrics.py
      backtest.py
    config.py
    run.py
  outputs/                          # Backtest outputs (equity curves, logs)
  README.md
  requirements.txt
```

## Quick Start
1) Install dependencies (ideally in a virtual environment):
```bash
pip install -r requirements.txt
```

2) Prepare your data as `./data/data.csv` with at least these columns:
```
date,ticker,open,high,low,close,volume,news,pe,roe
```
- `date` format: `YYYY-MM-DD`
- `news` can be a string (headline/summary concatenated per day)

3) Run:
```bash
python -m src.run --data ./data/data.csv --strategy rag_uncertainty --start 2019-01-01 --end 2024-12-31
```

## What’s Implemented
- **Data processing & feature engineering**: Missing value handling, returns/volatility/RSI features, prompt-ready features.
- **LLM TradingAgent**: Zero-shot / Few-shot / RAG strategies + uncertainty sampling. Simple `MockLLM` for offline runs.
- **RAG Retriever**: TF-IDF (if available) or fallback lexical similarity for news context.
- **Risk Management & Portfolio**: Confidence-weighted sizing, volatility targeting, max drawdown control, transaction costs.
- **Backtest Framework**: Time-based split, run & evaluate (CAGR, Sharpe, Max Drawdown) and save equity curve plot.

## Replace the LLM
- Implement and select `OpenAIModel` or `HFModel` in `src/agents/llm_interface.py` and pass `--llm openai` or `--llm hf`.
- Add your API key via environment variable or a config file.

## Fine-Tuning / PEFT (Stub)
- We provide a `FineTunedStubModel` demonstrating how to inject customized behavior; replace with your own PEFT/SFT/RLHF pipeline.

## Notes
- The dataset schema can vary; `data_loader` tries to be robust and will warn if some columns are missing.
- For true research, iterate on: prompt templates, retrieval windows, uncertainty calibration, portfolio construction.
