"""[Owner: TV3] Dựng model, lưu/đọc checkpoint — giao kèo chung giữa train (TV3) và evaluate/benchmark (TV2), app (TV4).

Model dùng khi suy luận nhận clip (B, 3, T, H, W) đã chuẩn hoá Kinetics (src/data/transforms.py):
- mode "cls": VideoModel(backbone, ClsHead) -> logits (B, 14); score bất thường = 1 - softmax[:, Normal].
- mode "mil": VideoModel(backbone, MILHead) -> LOGIT (B,) của xác suất bất thường; sigmoid(out) = điểm MIL.
  (MILHead kết thúc bằng Sigmoid; trả logit để khớp mặc định `evaluate.py --mil_output logit`.)

Định dạng checkpoint (torch.save dict):
    cls (weights/<model>_cls.pt): {"format": 1, "model": name, "mode": "cls", "classes": [...], "feat_dim": D,
                                   "state_dict": <VideoModel>, "epoch", "metrics", "cfg"}
    mil (weights/<model>_mil.pt): {"format": 1, "model": name, "mode": "mil", "feat_dim": D,
                                   "head_state_dict": <MILHead>, "backbone_ckpt": <đường dẫn cls ckpt | None>,
                                   "metrics", "cfg"}
    backbone_ckpt=None nghĩa là feature lấy từ backbone pretrained Kinetics (chưa fine-tune).
"""
from pathlib import Path

import torch
import torch.nn as nn

from src.data.constants import CLASSES
from src.models import build_model
from src.models.heads import ClsHead, MILHead
from src.utils.config import load_config

CKPT_FORMAT = 1


def load_cfg(path, overrides=None):
    return load_config(path, overrides)


def get_class_names(cfg=None):
    """Thứ tự lớp = chỉ số output của ClsHead (Normal ở vị trí 0)."""
    return list(CLASSES)


def normal_index(class_names, normal="NormalVideos"):
    return class_names.index(normal)


def apply_freeze(backbone, mode="none"):
    """Đóng băng theo cfg train.freeze nếu backbone hỗ trợ; "backbone" luôn làm được."""
    if hasattr(backbone, "set_freeze"):
        backbone.set_freeze(mode)
    elif mode == "backbone":
        backbone.requires_grad_(False)
    elif mode != "none":
        print(f"[cảnh báo] {type(backbone).__name__} không hỗ trợ freeze='{mode}', train toàn bộ")
    return backbone


class VideoModel(nn.Module):
    """backbone + head. Với MIL: mỗi clip là 1 snippet -> head nhận (B, 1, D)."""

    def __init__(self, backbone, head, mode="cls"):
        super().__init__()
        self.backbone, self.head, self.mode = backbone, head, mode

    def forward(self, x):
        f = self.backbone(x)
        if self.mode == "cls":
            return self.head(f)
        p = self.head(f[:, None, :])[:, 0]
        return torch.logit(p.float(), eps=1e-6)


def build_video_model(cfg, mode="cls", pretrained=None):
    m = cfg["model"]
    pretrained = m.get("pretrained", True) if pretrained is None else pretrained
    backbone, feat_dim = build_model(m["name"], cfg["data"]["num_classes"], pretrained=pretrained)
    if mode == "cls":
        head = ClsHead(feat_dim, cfg["data"]["num_classes"], p=m.get("dropout", 0.5))
    else:
        head = MILHead(feat_dim)
    return VideoModel(backbone, head, mode), feat_dim


def _torch_load(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


def save_checkpoint(path, model, cfg, mode="cls", **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = getattr(model, "_orig_mod", model)  # bỏ lớp bọc torch.compile nếu có
    ckpt = {"format": CKPT_FORMAT, "model": cfg["model"]["name"], "mode": mode, "classes": get_class_names(cfg),
            "cfg": cfg, **extra}
    if mode == "cls":
        ckpt["state_dict"] = model.state_dict()
        ckpt["feat_dim"] = model.head.fc[-1].in_features
    torch.save(ckpt, path)
    return path


def save_mil_checkpoint(path, head, cfg, feat_dim, backbone_ckpt=None, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": CKPT_FORMAT, "model": cfg["model"]["name"], "mode": "mil", "feat_dim": feat_dim,
                "head_state_dict": head.state_dict(), "backbone_ckpt": str(backbone_ckpt) if backbone_ckpt else None,
                "cfg": cfg, **extra}, path)
    return path


def load_backbone(cfg, ckpt=None, device="cpu"):
    """Backbone (không head) để trích feature: từ cls checkpoint, hoặc pretrained Kinetics nếu ckpt=None."""
    if ckpt is None:
        backbone, feat_dim = build_model(cfg["model"]["name"], cfg["data"]["num_classes"], pretrained=True)
    else:
        state = _torch_load(ckpt)
        if state.get("mode") != "cls":
            raise ValueError(f"{ckpt}: cần checkpoint chế độ cls để lấy backbone")
        model, feat_dim = build_video_model(cfg, "cls", pretrained=False)
        model.load_state_dict(state["state_dict"])
        backbone = model.backbone
    return backbone.to(device).eval(), feat_dim


def load_model(cfg, mode, ckpt, device="cpu"):
    """Model end-to-end ở chế độ eval cho evaluate.py / benchmark.py / app."""
    if ckpt is None:
        raise ValueError("Cần --ckpt")
    state = _torch_load(ckpt)
    if state.get("mode") != mode:
        raise ValueError(f"{ckpt} là checkpoint chế độ {state.get('mode')}, không phải {mode}")
    if state.get("model") != cfg["model"]["name"]:
        raise ValueError(f"{ckpt} là của model {state.get('model')}, config là {cfg['model']['name']}")
    if mode == "cls":
        model, _ = build_video_model(cfg, "cls", pretrained=False)
        model.load_state_dict(state["state_dict"])
    else:
        backbone, feat_dim = load_backbone(cfg, state.get("backbone_ckpt"), "cpu")
        head = MILHead(state["feat_dim"])
        head.load_state_dict(state["head_state_dict"])
        model = VideoModel(backbone, head, "mil")
    return model.to(device).eval()
