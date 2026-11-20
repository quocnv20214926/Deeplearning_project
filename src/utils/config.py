"""[Owner: TV3] Đọc yaml có kế thừa `_base_` + ghi đè từ dòng lệnh / biến môi trường.

    cfg = load_config("configs/m3_swin3d.yaml", ["train.batch_size=8", "data.frames_root=/kaggle/input/x"])

Biến môi trường CFG_OVERRIDES (các mục cách nhau bởi ";") được áp dụng cho MỌI lệnh đọc config
(train, evaluate, benchmark, prepare_data...) -> tiện trên Kaggle/Colab, không phải sửa base.yaml:
    export CFG_OVERRIDES="data.frames_root=/kaggle/input/ucf-crime-dataset;train.num_workers=4"
Thứ tự ưu tiên: yaml < CFG_OVERRIDES < overrides truyền vào (vd. --set).
"""
import os
from pathlib import Path

import yaml


def _merge(a, b):
    for k, v in b.items():
        a[k] = _merge(a.get(k, {}), v) if isinstance(v, dict) else v
    return a


def _load_yaml(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("_base_", None)
    return _merge(_load_yaml(path.parent / base), cfg) if base else cfg


def apply_overrides(cfg, overrides):
    """overrides: list["a.b.c=value"]; value được parse bằng yaml (số, bool, list...)."""
    for item in overrides or []:
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Override phải có dạng khoá=giá_trị, nhận được: {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _parse(value)
    return cfg


def _parse(value):
    v = yaml.safe_load(value)
    if isinstance(v, str):  # yaml 1.1 coi "5e-5" là chuỗi
        try:
            return float(v)
        except ValueError:
            pass
    return v


def env_overrides():
    return [s for s in os.environ.get("CFG_OVERRIDES", "").split(";") if s.strip()]


def load_config(path, overrides=None):
    cfg = _load_yaml(path)
    return apply_overrides(cfg, env_overrides() + list(overrides or []))
