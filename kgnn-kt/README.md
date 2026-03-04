# KGNN-KT (Open Source Reimplementation)

This repository provides a clean, review-friendly open-source implementation of **KGNN-KT** as described in the paper:

> **KGNN-KT: Enhancing Knowledge Tracing in Programming Education Through LLM-Extracted Knowledge Graphs**

The codebase is intentionally modular:
1. **LLM Knowledge Extraction + Graph Construction** (problem/code → normalized concepts → typed edges)
2. **Feature Encoding** (Problem text + code embeddings; student temporal state with LSTM; KG concepts with RGCN)
3. **Multimodal Fusion** (cross-modal attention gate + LayerNorm)
4. **Prediction Head + Training** (MLP + BCE, AdamW, early stopping)

> Note: This repo is designed to satisfy reproducibility / artifact-review expectations.
> It provides runnable scaffolding, sensible defaults, and clear interfaces. You can
> swap in your own datasets and/or LLM provider.

---

## Quick start

```bash
# 1) Create env
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install
pip install -r requirements.txt

# 3) (Optional) Build a demo KG from toy problems
python scripts/build_kg.py --input examples/toy_problems.jsonl --out_dir artifacts/toy_kg

# 4) Train on toy data (for smoke test)
python scripts/train.py --config examples/config_toy.yaml
```

---

## Repo layout

- `kgnn_kt/kg/`: graph schema, normalization, LLM extraction adapters
- `kgnn_kt/models/`: encoders, fusion, prediction head, full model
- `kgnn_kt/data/`: dataset schema + loaders (JSONL-based reference implementation)
- `scripts/`: CLI entrypoints
- `examples/`: toy dataset + config for reviewers
- `artifacts/`: generated outputs (ignored by git)

---

## Data format (reference)

This implementation expects **three** JSONL files:

### 1) problems.jsonl
Each line:
```json
{"problem_id":"p1","text":"...","code":"..."}
```

### 2) interactions.jsonl
Each line:
```json
{"user_id":"u1","problem_id":"p1","correct":1,"timestamp":1700000000}
```

### 3) user_profiles.jsonl (optional)
Each line:
```json
{"user_id":"u1","features":{"exp_months":12,"self_rating":3}}
```

---

## Knowledge graph schema

Node types:
- `problem`
- `concept` (data structure / algorithm / paradigm)

Edge types:
- `requires` (problem → concept, concept → concept)
- `subclass_of` (concept → concept)
- `antonym` (concept ↔ concept)

---

## LLM extraction

We provide a provider-agnostic adapter (`kgnn_kt/kg/llm.py`) with:
- `OpenAIChatProvider` (stubbed; reads `OPENAI_API_KEY`)
- `MockProvider` for offline testing

Prompt template follows the paper’s JSON extraction style.

---

## Citation

If you use this code, please cite the paper and this repository.

---

## License

MIT (see `LICENSE`).
