"""[Owner: TV2] Khoảng tin cậy bootstrap THEO VIDEO cho frame AUC + so sánh cặp giữa các model.
Mỗi lần lấy lại mẫu: chọn ngẫu nhiên (có hoàn lại) toàn bộ video test, tính AUC gộp trên các frame của chúng.
So sánh cặp: dùng CÙNG các mẫu lấy lại cho mọi model -> khác biệt AUC có khoảng tin cậy và p-value
(hai phía: 2 x tỉ lệ mẫu mà khác biệt ngược dấu / bằng 0).

    python -m src.bootstrap --names r2plus1d_18_cls i3d_r50_full_cls swin3d_t_cls --ref r2plus1d_18_cls
Đọc results/predictions/<name>.npz (khoá = tên video; do evaluate.py ghi) và nhãn test_frame_labels.npz.
Ghi results/tables/bootstrap.csv.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return {k: np.asarray(d[k]) for k in d.files}


def resample_indices(n_videos, n_boot=1000, seed=42):
    return np.random.default_rng(seed).integers(0, n_videos, size=(n_boot, n_videos))


def boot_auc(ys, ss, idx):
    out = np.full(len(idx), np.nan)
    for b, row in enumerate(idx):
        y = np.concatenate([ys[i] for i in row])
        if 0 < y.sum() < len(y):
            out[b] = roc_auc_score(y, np.concatenate([ss[i] for i in row]))
    return out


def ci(x, level=0.95):
    lo, hi = np.nanpercentile(x, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def paired_p(diffs):
    d = diffs[~np.isnan(diffs)]
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def bootstrap_report(scores_by_name, labels, n_boot=1000, seed=42, ref=None):
    """scores_by_name: {name: {video: điểm frame}}; labels: {video: nhãn frame}. Trả list dòng kết quả."""
    vids = sorted(labels)
    ys = [np.asarray(labels[v]).astype(int) for v in vids]
    idx = resample_indices(len(vids), n_boot, seed)
    point, boots = {}, {}
    for name, sc in scores_by_name.items():
        for v, y in zip(vids, ys):
            if v not in sc or len(sc[v]) != len(y):
                raise ValueError(f"{name}: thiếu/lệch độ dài video {v}")
        ss = [np.asarray(sc[v], float) for v in vids]
        point[name] = float(roc_auc_score(np.concatenate(ys), np.concatenate(ss)))
        boots[name] = boot_auc(ys, ss, idx)
    rows = []
    for name in scores_by_name:
        lo, hi = ci(boots[name])
        r = {"name": name, "auc": point[name], "ci_low": lo, "ci_high": hi, "ref": "", "diff": "",
             "diff_ci_low": "", "diff_ci_high": "", "p_value": ""}
        if ref and name != ref:
            d = boots[name] - boots[ref]
            dlo, dhi = ci(d)
            r.update({"ref": ref, "diff": point[name] - point[ref], "diff_ci_low": dlo,
                      "diff_ci_high": dhi, "p_value": paired_p(d)})
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True, help="tên file trong results/predictions (không .npz)")
    ap.add_argument("--ref", help="model tham chiếu cho so sánh cặp")
    ap.add_argument("--labels", default="data/processed/splits/test_frame_labels.npz")
    ap.add_argument("--pred_dir", default="results/predictions")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/tables/bootstrap.csv")
    args = ap.parse_args()
    if args.ref and args.ref not in args.names:
        ap.error("--ref phải nằm trong --names")

    labels = load_npz(args.labels)
    scores = {n: load_npz(Path(args.pred_dir) / f"{n}.npz") for n in args.names}
    rows = bootstrap_report(scores, labels, args.n_boot, args.seed, args.ref)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        line = f"{r['name']:<32} AUC {r['auc']:.4f}  95% CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        if r["ref"]:
            line += f"  | vs {r['ref']}: {r['diff']:+.4f} [{r['diff_ci_low']:+.4f}, {r['diff_ci_high']:+.4f}] p={r['p_value']:.3f}"
        print(line)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
