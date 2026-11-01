"""[Owner: TV3] Đọc yaml có kế thừa `_base_`."""
from pathlib import Path
import yaml


def _merge(a, b):
    for k, v in b.items():
        a[k] = _merge(a.get(k, {}), v) if isinstance(v, dict) else v
    return a


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("_base_", None)
    return _merge(load_config(path.parent / base), cfg) if base else cfg
