"""[Owner: TV2] Ablation freeze cho M2 I3D-R50 (tuần 3–4): với mỗi biến thể chạy
train (cls) -> evaluate -> benchmark  [+ với --mil: trích feature -> train_mil -> evaluate -> benchmark].

    python scripts/run_ablation.py --config configs/m2_i3d.yaml --no_wandb
    python scripts/run_ablation.py --config configs/m2_i3d.yaml --no_wandb --mil          # thêm chế độ MIL
    python scripts/run_ablation.py --config configs/m2_i3d.yaml --variants full=train.freeze=none --mil
    python scripts/run_ablation.py --config configs/m2_i3d.yaml --extra_set train.batch_size=8 train.grad_accum=2

Biến thể mặc định: full (train.freeze=none), partial (fine-tune vài block cuối), frozen (chỉ train head).
Biến thể "ten=khoa1=gt1,khoa2=gt2". Tag = <model.name>_<ten>:
    weights/<tag>_cls.pt, weights/<tag>_mil.pt, results/tables/<tag>_{cls,mil}.json
Feature MIL của mỗi biến thể nằm riêng ở data/features/<tag>/<model.name>/ (extract_features đặt tên thư mục theo
model.name nên nếu dùng chung sẽ lẫn feature giữa các biến thể).
Chạy lại an toàn: bước đã có kết quả được bỏ qua; train bị ngắt (chỉ có *_cls_last.pt) tự --resume.
--retrain: train lại cls (và trích lại feature, train lại MIL) từ đầu.
Lưu ý: train bị ngắt SAU khi đã có best.pt thì script coi là xong; hãy chạy tay `src.train --resume ...` rồi chạy lại.
Kết quả MIL do train_mil.py chấm từ feature được lưu thành <tag>_mil_features.json, rồi evaluate.py chấm lại
end-to-end (ghi đè <tag>_mil.json); hai AUC phải gần nhau, lệch nhiều là có lỗi.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.summary import collect_rows, keep_feature_result, needs_eval  # noqa: E402
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


def build_mil_commands(py, config, sets, tag, model, retrain=False, device=None, iterations=None):
    """(lệnh trích feature, lệnh train_mil) cho một biến thể. Feature riêng theo tag."""
    feats_root = f"data/features/{tag}"
    dev = ["--device", device] if device else []
    extract = [py, "-m", "src.extract_features", "--config", config, "--ckpt", f"weights/{tag}_cls.pt",
               "--out_root", feats_root, *dev, *sets]
    if retrain:
        extract.insert(-len(sets) if sets else len(extract), "--overwrite")
    train = [py, "-m", "src.train_mil", "--config", config, "--features", f"{feats_root}/{model}", "--tag", tag, *dev]
    if iterations:
        train += ["--iterations", str(iterations)]
    train += sets
    return extract, train


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--extra_set", nargs="*", default=[], help="override áp dụng cho mọi biến thể")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--mil", action="store_true", help="thêm chế độ MIL cho mỗi biến thể")
    ap.add_argument("--mil_iterations", type=int)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--device")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--skip_benchmark", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.extra_set)
    model = cfg["model"]["name"]
    py = sys.executable
    tables = ROOT / "results" / "tables"
    dev = ["--device", args.device] if args.device else []
    tags = []

    def bench(tag, mode, base):
        if args.skip_benchmark:
            return
        for prec in ("fp32", "fp16"):
            if (tables / f"{tag}_{mode}_{prec}_bench.json").exists():
                continue
            if run([py, "-m", "src.benchmark", *base, "--mode", mode, "--precision", prec, *dev]) and prec == "fp32":
                sys.exit(f"Benchmark lỗi ở {tag}/{mode}")   # fp16 lỗi (không có GPU) thì bỏ qua

    for name, ovr in parse_variants(args.variants):
        tag = f"{model}_{name}"
        tags.append(tag)
        sets = ["--set", *ovr, *args.extra_set]
        print(f"\n=== {tag}: {ovr}")
        # ---------------- cls ----------------
        best, last = ROOT / "weights" / f"{tag}_cls.pt", ROOT / "weights" / f"{tag}_cls_last.pt"
        if args.retrain or not best.exists():
            cmd = [py, "-m", "src.train", "--config", args.config, *sets, "--tag", tag]
            if args.epochs:
                cmd += ["--epochs", str(args.epochs)]
            if args.no_wandb:
                cmd.append("--no_wandb")
            cmd += dev
            if last.exists() and not args.retrain:
                cmd += ["--resume", str(last)]
            if run(cmd):
                sys.exit(f"Train lỗi ở {tag}")
        if needs_eval(tables / f"{tag}_cls.json") and run(
                [py, "-m", "src.evaluate", "--config", args.config, *sets, "--ckpt", str(best), "--tag", tag, *dev]):
            sys.exit(f"Evaluate lỗi ở {tag}")
        bench(tag, "cls", ["--config", args.config, *sets, "--ckpt", str(best), "--tag", tag, *dev])
        # ---------------- mil ----------------
        if args.mil:
            mil_ckpt = ROOT / "weights" / f"{tag}_mil.pt"
            if args.retrain or not mil_ckpt.exists():
                extract, train = build_mil_commands(py, args.config, sets, tag, model, args.retrain, args.device,
                                                    args.mil_iterations)
                if run(extract):
                    sys.exit(f"Trích feature lỗi ở {tag}")
                if run(train):
                    sys.exit(f"Train MIL lỗi ở {tag}")
            keep_feature_result(tables, tag)
            base = ["--config", args.config, *sets, "--ckpt", str(mil_ckpt), "--tag", tag, *dev]
            if needs_eval(tables / f"{tag}_mil.json") and run([py, "-m", "src.evaluate", *base, "--mode", "mil"]):
                sys.exit(f"Evaluate MIL lỗi ở {tag}")
            fp, ep = tables / f"{tag}_mil_features.json", tables / f"{tag}_mil.json"
            if fp.exists() and ep.exists():
                a, b = json.loads(fp.read_text())["frame_auc"], json.loads(ep.read_text())["frame_auc"]
                print(f"MIL {tag}: AUC từ feature {a:.4f} vs end-to-end {b:.4f} (chênh {abs(a - b):.4f})")
            bench(tag, "mil", ["--config", args.config, *sets, "--ckpt", str(mil_ckpt), "--tag", tag, *dev])

    rows = [r for r in collect_rows(tables) if r["model"] in tags and r["precision"] == "fp32"]
    print("\n=== Kết quả ablation (fp32)")
    print(f"{'tag':<22}{'mode':>5}{'AUC':>8}{'FPR(Normal)':>13}{'params(M)':>11}{'VRAM(MB)':>10}{'FPS':>8}")
    f = lambda x, p=3: "-" if x is None else f"{x:.{p}f}"  # noqa: E731
    for r in rows:
        print(f"{r['model']:<22}{r['mode']:>5}{f(r['frame_auc']):>8}{f(r['normal_video_fpr']):>13}"
              f"{f(r['params_m'], 1):>11}{f(r['vram_peak_mb'], 0):>10}{f(r['fps'], 0):>8}")


if __name__ == "__main__":
    main()
