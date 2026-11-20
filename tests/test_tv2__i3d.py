import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorchvideo")

from src.models.i3d import build_i3d  # noqa: E402


def _model():
    m, d = build_i3d(14, pretrained=False)   # không cần tải trọng số
    return m, d


def test_output_shape_matches_protocol_input():
    m, d = _model()
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 16, 112, 112))
    assert d == 2048 and out.shape == (2, 2048)


def test_freeze_modes():
    m, _ = _model()
    total = sum(p.numel() for p in m.parameters())
    m.set_freeze("backbone")
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 0
    m.set_freeze("partial")
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert 0 < trainable < total
    assert all(not p.requires_grad for p in m.stages[0].parameters())
    assert all(p.requires_grad for p in m.stages[4].parameters())
    m.set_freeze("none")
    assert all(p.requires_grad for p in m.parameters())


def test_frozen_stages_stay_in_eval_mode_during_train():
    m, _ = _model()
    m.set_freeze("partial")
    m.train()
    assert not m.stages[0].training and not m.stages[2].training and m.stages[4].training
