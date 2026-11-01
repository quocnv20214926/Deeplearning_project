"""[Owner: TV2] Metric dùng chung: frame-level AUC, FPR, AUC theo lớp."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def frame_auc(scores: dict, labels: dict):
    """AUC gộp toàn bộ frame của mọi video test. scores/labels: {video: np.array}."""
    vids = sorted(labels)
    y = np.concatenate([labels[v] for v in vids])
    s = np.concatenate([scores[v] for v in vids])
    return roc_auc_score(y, s)


def per_class_auc(scores, labels, video_class: dict, normal="Normal"):
    """AUC từng lớp: frame của lớp đó + toàn bộ frame video Normal."""
    normals = [v for v in labels if video_class[v].startswith(normal)]
    out = {}
    for c in sorted(set(video_class.values())):
        if c.startswith(normal):
            continue
        vids = [v for v in labels if video_class[v] == c] + normals
        out[c] = frame_auc({v: scores[v] for v in vids}, {v: labels[v] for v in vids})
    return out


def false_positive_rate(scores, labels, threshold=0.5):
    """FPR = FP / (FP + TN) trên toàn bộ frame có nhãn 0."""
    y = np.concatenate([labels[v] for v in sorted(labels)])
    s = np.concatenate([scores[v] for v in sorted(labels)])
    neg = y == 0
    return float((s[neg] >= threshold).mean())


def fpr_at_tpr(scores, labels, tpr_target=0.9):
    y = np.concatenate([labels[v] for v in sorted(labels)])
    s = np.concatenate([scores[v] for v in sorted(labels)])
    fpr, tpr, _ = roc_curve(y, s)
    return float(fpr[np.searchsorted(tpr, tpr_target)])


def snippet_to_frame(snippet_scores, num_frames):
    """Trải điểm snippet/segment ra từng frame (nội suy hằng từng đoạn)."""
    idx = np.linspace(0, len(snippet_scores), num_frames, endpoint=False).astype(int)
    return np.asarray(snippet_scores)[np.clip(idx, 0, len(snippet_scores) - 1)]
