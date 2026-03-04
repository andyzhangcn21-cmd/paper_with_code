#!/usr/bin/env python
from __future__ import annotations
import argparse, yaml, os
import torch
from kgnn_kt.config import AppConfig
from kgnn_kt.train import _device, _build_kg_inputs, _dummy_student_tensors
from kgnn_kt.data.dataset import load_data, KTExampleDataset, collate_batch
from kgnn_kt.kg.build import load_graph
from kgnn_kt.models.model import KGNNKT
from torch.utils.data import DataLoader
from kgnn_kt.utils.metrics import auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config (same as train)")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint (best.pt)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = AppConfig.model_validate(yaml.safe_load(f))

    device = _device(cfg.train.device)
    data = load_data(cfg.data.problems_path, cfg.data.interactions_path, cfg.data.user_profiles_path)
    ds = KTExampleDataset(data.interactions)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_batch)

    graph = load_graph(os.path.join(cfg.data.kg_dir, "graph.json")) if cfg.data.kg_dir else None

    model = KGNNKT(
        text_model_name=cfg.model.text_model_name,
        code_model_name=cfg.model.code_model_name,
        concept_in_dim=cfg.model.concept_dim_in,
        rgcn_hidden=cfg.model.rgcn_hidden,
        fused_dim=cfg.model.fused_dim,
        dropout=cfg.model.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    ys, yh = [], []
    with torch.no_grad():
        for batch in loader:
            y = batch["y"].to(device)
            problem_texts = [data.problems[pid].text for pid in batch["problem_id"]]
            problem_codes = [data.problems[pid].code for pid in batch["problem_id"]]
            student_events, student_profiles = _dummy_student_tensors(len(y), device=device)
            if graph:
                kg_inputs = _build_kg_inputs(graph, batch["problem_id"], device)
            else:
                kg_inputs = None
            yhat = model(problem_texts, problem_codes, student_events, student_profiles, kg_inputs) if kg_inputs else torch.zeros_like(y)
            ys.extend(y.detach().cpu().numpy().tolist())
            yh.extend(yhat.detach().cpu().numpy().tolist())

    print({"auc": auc(ys, yh), "n": len(ys)})

if __name__ == "__main__":
    main()
