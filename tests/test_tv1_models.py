"""[Owner: TV1] Test M1 R(2+1)D-18 và B0 MobileNetV3-GRU.

    pytest tests/                       # test nhanh (trọng số ngẫu nhiên, CPU)
    RUN_SLOW=1 pytest tests/            # + tải trọng số pretrained, train thử trên GPU với clip thật
"""
import os

import pytest
import torch

from src.data.transforms import KINETICS_MEAN, KINETICS_STD
from src.models import build_model
from src.models.heads import ClsHead
from src.models.mobilenet_gru import IMAGENET_MEAN, IMAGENET_STD, MobileNetGRU
from src.models.r2plus1d import R2Plus1DBackbone

TV1_MODELS = {"r2plus1d_18": 512, "mobilenet_gru": 256}
slow = pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="đặt RUN_SLOW=1 để chạy")


@pytest.mark.parametrize("name,dim", TV1_MODELS.items())
def test_backbone_and_head(name, dim):
    backbone, feat_dim = build_model(name, num_classes=14, pretrained=False)
    assert feat_dim == dim
    x = torch.randn(2, 3, 16, 112, 112)
    with torch.no_grad():
        f = backbone.eval()(x)
        assert f.shape == (2, dim)
        assert ClsHead(feat_dim, 14)(f).shape == (2, 14)
        assert backbone(torch.randn(1, 3, 8, 112, 112)).shape == (1, dim)  # clip 8 frame (ablation)


def test_mobilenet_renormalisation():
    """Đầu vào chuẩn hoá Kinetics phải thành đúng ảnh chuẩn hoá ImageNet."""
    m = MobileNetGRU(pretrained=False)
    pixels = torch.rand(1, 3, 8, 8)
    k_mean, k_std = (torch.tensor(v).view(1, 3, 1, 1) for v in (KINETICS_MEAN, KINETICS_STD))
    i_mean, i_std = (torch.tensor(v).view(1, 3, 1, 1) for v in (IMAGENET_MEAN, IMAGENET_STD))
    got = (pixels - k_mean) / k_std * m.scale + m.shift
    assert torch.allclose(got, (pixels - i_mean) / i_std, atol=1e-5)


def test_r2plus1d_freeze():
    m = R2Plus1DBackbone(pretrained=False).set_freeze("partial").train()
    assert not any(p.requires_grad for s in (m.net.stem, m.net.layer1, m.net.layer2) for p in s.parameters())
    assert all(p.requires_grad for p in m.net.layer3.parameters())
    assert not m.net.layer2.training and m.net.layer3.training
    m.set_freeze("none")
    assert all(p.requires_grad for p in m.parameters())


def test_mobilenet_freeze():
    m = MobileNetGRU(pretrained=False).set_freeze("backbone").train()
    assert not any(p.requires_grad for p in m.cnn.parameters()) and not m.cnn.training
    assert all(p.requires_grad for p in m.gru.parameters())


@slow
@pytest.mark.parametrize("name", TV1_MODELS)
def test_pretrained_weights(name):
    backbone, _ = build_model(name, num_classes=14, pretrained=True)
    with torch.no_grad():
        assert torch.isfinite(backbone.eval()(torch.randn(1, 3, 16, 112, 112))).all()


@slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cần GPU")
@pytest.mark.parametrize("name", TV1_MODELS)
def test_overfit_real_clips(name):
    """Dữ liệu thật + backbone + ClsHead học thuộc được 8 clip (kiểm tra cả pipeline)."""
    from src.data.transforms import ClipTransform
    from src.data.ucf_crime import UCFCrimeClips, data_paths
    from src.utils.config import load_config

    cfg = load_config("configs/base.yaml")
    if not (data_paths(cfg)["split_dir"] / "train.txt").exists():
        pytest.skip("chưa chạy scripts/prepare_data.py")
    ds = UCFCrimeClips(cfg, "train", ClipTransform(112))
    normal = [i for i, it in enumerate(ds.items) if it[2] == 0][:4]
    anomaly = [i for i, it in enumerate(ds.items) if it[2] > 0][:4]
    batch = [ds[i] for i in normal + anomaly]
    x = torch.stack([b[0] for b in batch]).cuda()
    y = torch.tensor([b[1] for b in batch]).cuda()

    torch.manual_seed(0)
    backbone, feat_dim = build_model(name, num_classes=14, pretrained=True)
    model = torch.nn.Sequential(backbone, ClsHead(feat_dim, 14, p=0.0)).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    for _ in range(40):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.2 < losses[0], losses
