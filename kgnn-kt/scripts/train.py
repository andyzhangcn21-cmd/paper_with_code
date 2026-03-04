#!/usr/bin/env python
from __future__ import annotations
import argparse, yaml
from kgnn_kt.config import AppConfig
from kgnn_kt.train import train_from_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = AppConfig.model_validate(yaml.safe_load(f))
    train_from_config(cfg)

if __name__ == "__main__":
    main()
