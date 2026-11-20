"""[Owner: TV2] Ablation freeze cho M2 I3D-R50 (tuần 3–4): với mỗi biến thể chạy train -> evaluate -> benchmark.

    python scripts/run_ablation.py --config configs/m2_i3d.yaml --no_wandb
    python scripts/run_ablation.py --config configs/m2_i3d.yaml --variants full=train.freeze=none partial=train.freeze=partial
    python scripts/run_ablation.py --config configs/m2_i3d.yaml --extra_set train.batch_size=8 train.grad_accum=2

Biến thể mặc định: full (train.freeze=none), partial (fine-tune vài block cuối), frozen (chỉ train head).
Biến thể "ten=khoa1=gt1,khoa2=gt2". Tag = <model.name>_<ten> -> weights/<tag>_cls.pt, results/tables/<tag>_cls.json.
Chạy lại an toàn: bước đã có kết quả được bỏ qua. Nếu train bị ngắt (chỉ có *_cls_last.pt) sẽ tự --resume.
Lưu ý: nếu train bị ngắt SAU khi đã có best.pt, script coi là xong; hãy chạy tay `src.train --resume ...` rồi chạy lại.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.summary import collect_rows  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DEFAULT_VARIANTS = ["full=train.freeze=none", "partial=train.freeze=partial", "frozen=train.freeze=backbone"]


def parse_variants(specs):
    out = []
    for s in specs:
        name, _, ovr = s.partition("=")
        if not name or not ovr:
            raise ValueError(f"Biến thể phải có dạng ten=khoa=gt[,khoa=gt]: {s!r}")
        out.append((name, [o for o in ovr.split(",") if o]))
    return out


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--extra_set", nargs="*", default=[], help="override áp dụng cho mọi biến thể")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--device")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--skip_benchmark", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.extra_set)
    model = cfg["model"]["name"]
    py = sys.executable
    tags = []
    for name, ovr in parse_variants(args.variants):
        tag = f"{model}_{name}"
        tags.append(tag)
        sets = ["--set", *ovr, *args.extra_set]
        best, last = ROOT / "weights" / f"{tag}_cls.pt", ROOT / "weights" / f"{tag}_cls_last.pt"
        print(f"\n=== {tag}: {ovr}")
        if args.retrain or not best.exists():
            cmd = [py, "-m", "src.train", "--config", args.config, *sets, "--tag", tag]
            if args.epochs:
                cmd += ["--epochs", str(args.epochs)]
            if args.no_wandb:
                cmd.append("--no_wandb")
            if args.device:
                cmd += ["--device", args.device]
            if last.exists() and not args.retrain:
                cmd += ["--resume", str(last)]
            if run(cmd):
                sys.exit(f"Train lỗi ở {tag}")
        ev = ROOT / "results" / "tables" / f"{tag}_cls.json"
        base = ["--config", args.config, *sets, "--ckpt", str(best), "--tag", tag]
        dev = ["--device", args.device] if args.device else []
        if not ev.exists() and run([py, "-m", "src.evaluate", *base, *dev]):
            sys.exit(f"Evaluate lỗi ở {tag}")
        if not args.skip_benchmark:
            for prec in ("fp32", "fp16"):
                if (ROOT / "results" / "tables" / f"{tag}_cls_{prec}_bench.json").exists():
                    continue
                if run([py, "-m", "src.benchmark", *base, "--precision", prec, *dev]) and prec == "fp32":
                    sys.exit(f"Benchmark lỗi ở {tag}")   # fp16 lỗi (không có GPU) thì bỏ qua

    rows = [r for r in collect_rows(ROOT / "results" / "tables") if r["model"] in tags and r["precision"] == "fp32"]
    print("\n=== Kết quả ablation (fp32)")
    print(f"{'tag':<24}{'AUC':>8}{'FPR(Normal)':>13}{'params(M)':>11}{'VRAM(MB)':>10}{'FPS':>8}")
    f = lambda x, p=3: "-" if x is None else f"{x:.{p}f}"  # noqa: E731
    for r in rows:
        print(f"{r['model']:<24}{f(r['frame_auc']):>8}{f(r['normal_video_fpr']):>13}"
              f"{f(r['params_m'], 1):>11}{f(r['vram_peak_mb'], 0):>10}{f(r['fps'], 0):>8}")


if __name__ == "__main__":
    main()
