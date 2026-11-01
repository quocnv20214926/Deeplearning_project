"""[Owner: TV3 — framework; mỗi TV chạy cho model của mình]
Fine-tune backbone + ClsHead (chế độ cls).
    python -m src.train --config configs/m1_r2plus1d.yaml
TODO(TV3): vòng train AMP, AdamW + cosine, val mỗi epoch (frame AUC), lưu best -> weights/<model>_cls.pt, log wandb.
"""
import argparse
from src.utils.config import load_config


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    raise NotImplementedError(f"TV3: training loop for {cfg['model']['name']}")


if __name__ == "__main__":
    main()
