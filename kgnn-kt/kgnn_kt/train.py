from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .config import AppConfig
from .utils.seed import set_seed
from .utils.logging import JsonlLogger
from .utils.metrics import auc
from .data.dataset import load_data, KTExampleDataset, collate_batch
from .kg.build import load_graph
from .models.model import KGNNKT, KGInputs

def _device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def _dummy_student_tensors(batch_size: int, device: torch.device):
    # For artifact review we provide a deterministic toy featurizer.
    # events: (B, T, 32), profiles: (B, 32)
    T = 10
    events = torch.zeros((batch_size, T, 32), device=device)
    profiles = torch.zeros((batch_size, 32), device=device)
    return events, profiles

def _build_kg_inputs(graph, batch_problem_ids: List[str], device: torch.device) -> KGInputs:
    idx = graph.node_index()
    # Build minimal tensors:
    # - node features: random for concept nodes; zeros for problem nodes (placeholder)
    N = len(graph.nodes)
    D0 = 100
    node_feat = torch.zeros((N, D0), device=device)

    # simple initialization for concept nodes
    for i, n in enumerate(graph.nodes):
        if n.node_type == "concept":
            torch.manual_seed(abs(hash(n.label)) % (2**31 - 1))
            node_feat[i] = torch.randn(D0, device=device) * 0.1

    # adjacency per relation
    rel_map = {"subclass_of": 0, "requires": 1, "antonym": 2}
    adjs = torch.zeros((3, N, N), device=device)
    for e in graph.edges:
        r = rel_map[e.edge_type]
        if e.src in idx and e.dst in idx:
            adjs[r, idx[e.src], idx[e.dst]] = 1.0

    # per-sample problem node index
    p_nodes = []
    for pid in batch_problem_ids:
        node_id = f"problem:{pid}"
        p_nodes.append(idx.get(node_id, 0))
    return KGInputs(node_features=node_feat, adjs=adjs, problem_node_ids=p_nodes)

def train_from_config(cfg: AppConfig) -> None:
    set_seed(cfg.train.seed)
    device = _device(cfg.train.device)
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    logger = JsonlLogger(os.path.join(cfg.train.output_dir, "metrics.jsonl"))

    data = load_data(cfg.data.problems_path, cfg.data.interactions_path, cfg.data.user_profiles_path)
    ds = KTExampleDataset(data.interactions)

    n = len(ds)
    n_val = max(1, int(0.1 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.train.seed))

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_batch)

    graph = None
    if cfg.data.kg_dir:
        graph = load_graph(os.path.join(cfg.data.kg_dir, "graph.json"))

    model = KGNNKT(
        text_model_name=cfg.model.text_model_name,
        code_model_name=cfg.model.code_model_name,
        concept_in_dim=cfg.model.concept_dim_in,
        rgcn_hidden=cfg.model.rgcn_hidden,
        fused_dim=cfg.model.fused_dim,
        dropout=cfg.model.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    bce = nn.BCELoss()

    best = -1e9
    bad = 0

    for epoch in range(1, cfg.train.max_epochs + 1):
        model.train()
        tr_losses = []
        for batch in train_loader:
            y = batch["y"].to(device)
            probs = []

            # Assemble text/code batch
            problem_texts = [data.problems[pid].text for pid in batch["problem_id"]]
            problem_codes = [data.problems[pid].code for pid in batch["problem_id"]]

            student_events, student_profiles = _dummy_student_tensors(len(y), device=device)
            kg_inputs = _build_kg_inputs(graph, batch["problem_id"], device) if graph else _build_kg_inputs(
                # fallback: empty graph with only problems (toy)
                _toy_graph_from_problem_ids(batch["problem_id"]), batch["problem_id"], device
            )

            yhat = model(problem_texts, problem_codes, student_events, student_profiles, kg_inputs)
            loss = bce(yhat, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            opt.step()
            tr_losses.append(float(loss.detach().cpu()))

        # Validation
        model.eval()
        ys, yh = [], []
        with torch.no_grad():
            for batch in val_loader:
                y = batch["y"].to(device)
                problem_texts = [data.problems[pid].text for pid in batch["problem_id"]]
                problem_codes = [data.problems[pid].code for pid in batch["problem_id"]]
                student_events, student_profiles = _dummy_student_tensors(len(y), device=device)
                kg_inputs = _build_kg_inputs(graph, batch["problem_id"], device) if graph else _build_kg_inputs(
                    _toy_graph_from_problem_ids(batch["problem_id"]), batch["problem_id"], device
                )
                yhat = model(problem_texts, problem_codes, student_events, student_profiles, kg_inputs)
                ys.extend(y.detach().cpu().numpy().tolist())
                yh.extend(yhat.detach().cpu().numpy().tolist())

        val_auc = auc(ys, yh)
        payload = {"epoch": epoch, "train_loss": float(np.mean(tr_losses)), "val_auc": float(val_auc)}
        logger.log(payload)
        print(payload)

        if np.isnan(val_auc):
            # if degenerate labels in toy split
            val_auc = -1.0

        if val_auc > best:
            best = val_auc
            bad = 0
            torch.save(model.state_dict(), os.path.join(cfg.train.output_dir, "best.pt"))
        else:
            bad += 1
            if bad >= cfg.train.patience:
                print(f"Early stopping at epoch={epoch} (best val_auc={best:.4f})")
                break

def _toy_graph_from_problem_ids(problem_ids: List[str]):
    from .kg.schema import Graph, Node, Edge
    g = Graph()
    for pid in set(problem_ids):
        g.nodes.append(Node(node_id=f"problem:{pid}", node_type="problem", label=pid))
    return g
