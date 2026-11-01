"""Model registry. Mỗi thành viên đăng ký backbone của mình tại đây."""
from .r2plus1d import build_r2plus1d      # TV1
from .mobilenet_gru import build_mobilenet_gru  # TV1
from .i3d import build_i3d                # TV2
from .swin3d import build_swin3d          # TV3
from .x3d import build_x3d                # TV4

REGISTRY = {
    "r2plus1d_18": build_r2plus1d,
    "mobilenet_gru": build_mobilenet_gru,
    "i3d_r50": build_i3d,
    "swin3d_t": build_swin3d,
    "x3d_s": build_x3d,
}


def build_model(name: str, num_classes: int, pretrained: bool = True):
    """Trả về (backbone, feat_dim). Head gắn riêng trong heads.py."""
    return REGISTRY[name](num_classes=num_classes, pretrained=pretrained)
