"""[Owner: TV2] Chạy evaluate + benchmark cho MỌI checkpoint trong weights/, rồi gộp results/tables/summary.csv.

    python scripts/run_all_benchmarks.py                       # tất cả: evaluate + fp32/fp16/onnx
    python scripts/run_all_benchmarks.py --only i3d r2plus     # chỉ checkpoint có tên chứa chuỗi này
    python scripts/run_all_benchmarks.py --precisions fp32 --no_eval
    python scripts/run_all_benchmarks.py --summary_only        # chỉ dựng lại summary.csv từ JSON có sẵn

Tự tìm weights/*_cls.pt và weights/*_mil.pt; cấu hình lấy từ chính checkpoint (cfg lúc train) nên khớp với
model đã train. Nếu checkpoint train ở máy khác (đường dẫn dữ liệu khác) thêm `--set data.frames_root=...`.
Mỗi benchmark chạy trong process riêng (RAM/VRAM sạch). Bước đã có kết quả sẽ được bỏ qua (dùng --force để chạy lại).
fp16 chỉ chạy khi có GPU; onnx chỉ chạy khi có <onnx_dir>/<tag>_<mode>.onnx (TV4 xuất từ export_onnx.py).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.summary import build_summary  # noqa: E402


def discover(weights_dir, only):
    import torch
    import yaml

    tmp = ROOT / "results" / "tmp_cfg"
    tmp.mkdir(parents=True, exist_ok=True)
    runs = []
    for p in sorted(Path(weights_dir).glob("*_cls.pt")) + sorted(Path(weights_dir).glob("*_mil.pt")):
        if only and not any(o in p.stem for o in only):
            continue
        state = torch.load(p, map_location="cpu", weights_only=False)
        mode = state["mode"]
        tag = p.stem[: -len(f"_{mode}")]
        cfg_path = tmp / f"{tag}_{mode}.yaml"
        cfg_path.write_text(yaml.safe_dump(json.loads(json.dumps(state["cfg"])), allow_unicode=True),
                            encoding="utf-8")
        runs.append({"tag": tag, "mode": mode, "ckpt": str(p), "cfg": str(cfg_path)})
    return runs


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "onnx"], choices=["fp32", "fp16", "onnx"])
    ap.add_argument("--onnx_pattern", default="weights/onnx/{tag}_{mode}.onnx")
    ap.add_argument("--set", nargs="*", default=[], metavar="KHOA=GIA_TRI")
    ap.add_argument("--device")
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary_only", action="store_true")
    args = ap.parse_args()

    tables = ROOT / args.tables
    failures = []
    if not args.summary_only:
        import torch

        has_gpu = (args.device or "").startswith("cuda") or (not args.device and torch.cuda.is_available())
        extra = (["--set", *args.set] if args.set else []) + (["--device", args.device] if args.device else [])
        runs = discover(ROOT / args.weights, args.only)
        if not runs:
            raise SystemExit(f"Không thấy checkpoint *_cls.pt / *_mil.pt trong {args.weights}")
        print(f"{len(runs)} checkpoint: {[r['tag'] + '/' + r['mode'] for r in runs]}")
        for r in runs:
            base = ["--config", r["cfg"], "--ckpt", r["ckpt"], "--mode", r["mode"], "--tag", r["tag"]]
            if not args.no_eval and (args.force or not (tables / f"{r['tag']}_{r['mode']}.json").exists()):
                if run([sys.executable, "-m", "src.evaluate", *base, *extra]):
                    failures.append(f"evaluate {r['tag']}/{r['mode']}")
            for prec in args.precisions:
                if (tables / f"{r['tag']}_{r['mode']}_{prec}_bench.json").exists() and not args.force:
                    continue
                cmd = [sys.executable, "-m", "src.benchmark", *base, "--precision", prec, *extra]
                if prec == "fp16" and not has_gpu:
                    print(f"bỏ qua fp16 {r['tag']}/{r['mode']}: không có GPU")
                    continue
                if prec == "onnx":
                    onnx = ROOT / args.onnx_pattern.format(tag=r["tag"], mode=r["mode"])
                    if not onnx.exists():
                        print(f"bỏ qua onnx {r['tag']}/{r['mode']}: chưa có {onnx}")
                        continue
                    cmd += ["--onnx", str(onnx)]
                if run(cmd):
                    failures.append(f"benchmark {prec} {r['tag']}/{r['mode']}")

    rows = build_summary(tables, tables / "summary.csv")
    print(f"\nsummary.csv: {len(rows)} dòng -> {tables / 'summary.csv'}")
    if failures:
        print("LỖI ở các bước:", *failures, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
