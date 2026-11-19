"""[Owner: TV2] Metric dùng chung: frame-level AUC, FPR, AUC theo lớp, ngưỡng cảnh báo.

Quy ước: scores/labels là dict {video_name: np.ndarray 1-D}, độ dài mỗi video
phải BẰNG NHAU giữa scores và labels (đơn vị: frame đã lấy mẫu 1/10 của Kaggle).
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _concat(scores: dict, labels: dict, vids=None):
    """Nối nhãn/điểm theo danh sách video (cho phép lặp -> dùng cho bootstrap).
    Kiểm tra độ dài khớp để không bị sai số âm thầm."""
    vids = sorted(labels) if vids is None else list(vids)
    for v in vids:
        if v not in scores:
            raise KeyError(f"Thiếu điểm dự đoán cho video: {v}")
        if len(scores[v]) != len(labels[v]):
            raise ValueError(
                f"{v}: {len(scores[v])} scores vs {len(labels[v])} labels (lệch độ dài)"
            )
    y = np.concatenate([np.asarray(labels[v]) for v in vids]).astype(int)
    s = np.concatenate([np.asarray(scores[v]) for v in vids]).astype(float)
    return y, s


def _check_two_classes(y):
    if len(np.unique(y)) < 2:
        raise ValueError("Cần cả frame bình thường (0) và bất thường (1) để tính AUC/ROC")


# --------------------------------------------------------------------------- #
# Frame-level metrics
# --------------------------------------------------------------------------- #
def frame_auc(scores: dict, labels: dict, vids=None) -> float:
    """AUC gộp toàn bộ frame của mọi video test."""
    y, s = _concat(scores, labels, vids)
    _check_two_classes(y)
    return float(roc_auc_score(y, s))


def per_class_auc(scores, labels, video_class: dict, normal="Normal") -> dict:
    """AUC từng lớp bất thường: frame của lớp đó + toàn bộ frame video Normal.
    Chỉ xét lớp có mặt trong `labels` (tránh lỗi khi video_class chứa cả train)."""
    normals = [v for v in labels if video_class[v].startswith(normal)]
    classes = sorted({video_class[v] for v in labels})
    out = {}
    for c in classes:
        if c.startswith(normal):
            continue
        vids = [v for v in labels if video_class[v] == c] + normals
        y, _ = _concat(scores, labels, vids)
        out[c] = frame_auc(scores, labels, vids) if len(np.unique(y)) == 2 else float("nan")
    return out


def false_positive_rate(scores, labels, threshold=0.5, vids=None) -> float:
    """FPR = FP / (FP + TN) trên toàn bộ frame có nhãn 0."""
    y, s = _concat(scores, labels, vids)
    neg = y == 0
    if not neg.any():
        raise ValueError("Không có frame nhãn 0")
    return float((s[neg] >= threshold).mean())


def normal_video_fpr(scores, labels, video_class, threshold=0.5, normal="Normal") -> float:
    """FPR chỉ tính trên video Normal (sát với 'báo động giả' khi triển khai)."""
    vids = [v for v in labels if video_class[v].startswith(normal)]
    return false_positive_rate(scores, labels, threshold, vids)


def roc_points(scores, labels, vids=None):
    """Trả (fpr, tpr, thresholds) để vẽ ROC trong visualize.py."""
    y, s = _concat(scores, labels, vids)
    _check_two_classes(y)
    return roc_curve(y, s)


def fpr_at_tpr(scores, labels, tpr_target=0.9, vids=None) -> float:
    """FPR nhỏ nhất đạt TPR >= tpr_target."""
    fpr, tpr, _ = roc_points(scores, labels, vids)
    return float(fpr[np.searchsorted(tpr, tpr_target, side="left")])


def threshold_at_fpr(scores, labels, target_fpr=0.01, vids=None) -> float:
    """Ngưỡng thấp nhất sao cho FPR <= target_fpr (TV3 dùng trên tập val → TV4)."""
    fpr, _, thr = roc_points(scores, labels, vids)
    idx = int(np.searchsorted(fpr, target_fpr, side="right") - 1)
    t = float(thr[max(idx, 0)])
    if not np.isfinite(t):  # điểm đầu của roc_curve có thể là inf
        _, s = _concat(scores, labels, vids)
        t = float(s.max()) + 1e-6
    return t


# --------------------------------------------------------------------------- #
# Phân loại 14 lớp (video-level)
# --------------------------------------------------------------------------- #
def multiclass_accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true và y_pred khác độ dài")
    return float((y_true == y_pred).mean())


# --------------------------------------------------------------------------- #
# Chuyển điểm clip/segment -> điểm từng frame
# --------------------------------------------------------------------------- #
def snippet_to_frame(snippet_scores, num_frames) -> np.ndarray:
    """MIL: trải điểm segment đều ra toàn video (hằng số từng đoạn)."""
    snippet_scores = np.asarray(snippet_scores)
    idx = np.linspace(0, len(snippet_scores), num_frames, endpoint=False).astype(int)
    return snippet_scores[np.clip(idx, 0, len(snippet_scores) - 1)]


def clips_to_frame(clip_scores, num_frames, clip_len=16, stride=8) -> np.ndarray:
    """Cls: clip k phủ frame [k*stride, k*stride+clip_len). Điểm frame = trung bình
    các clip phủ nó; frame không clip nào phủ (đuôi video) lấy điểm clip gần nhất."""
    clip_scores = np.asarray(clip_scores, dtype=float)
    acc = np.zeros(num_frames)
    cnt = np.zeros(num_frames)
    for k, sc in enumerate(clip_scores):
        a, b = k * stride, min(k * stride + clip_len, num_frames)
        if a >= num_frames:
            break
        acc[a:b] += sc
        cnt[a:b] += 1
    out = np.empty(num_frames)
    covered = cnt > 0
    out[covered] = acc[covered] / cnt[covered]
    if not covered.all():
        if not covered.any():
            raise ValueError("Không có clip nào phủ frame")
        ci = np.flatnonzero(covered)
        miss = np.flatnonzero(~covered)
        out[miss] = out[ci[np.abs(miss[:, None] - ci[None, :]).argmin(axis=1)]]
    return out


# --------------------------------------------------------------------------- #
# Video-level (cross-dataset, vd RWF-2000: chỉ có nhãn cả clip)
# --------------------------------------------------------------------------- #
def video_scores(scores: dict, agg="max", k=3) -> dict:
    """Gộp điểm frame -> 1 điểm/video. agg: 'max' | 'topk_mean' | 'mean'."""
    out = {}
    for v, s in scores.items():
        s = np.asarray(s, dtype=float)
        if agg == "max":
            out[v] = float(s.max())
        elif agg == "mean":
            out[v] = float(s.mean())
        elif agg == "topk_mean":
            out[v] = float(np.sort(s)[-min(k, len(s)):].mean())
        else:
            raise ValueError(f"agg không hợp lệ: {agg}")
    return out


def video_auc(scores: dict, video_labels: dict, agg="max", k=3) -> float:
    """AUC mức video. video_labels: {video: 0/1}."""
    vs = video_scores(scores, agg, k)
    vids = sorted(video_labels)
    y = np.array([video_labels[v] for v in vids], dtype=int)
    _check_two_classes(y)
    return float(roc_auc_score(y, [vs[v] for v in vids]))
