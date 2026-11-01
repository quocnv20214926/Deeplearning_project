"""[Owner: TV2] Đo hiệu năng: params, file size (MB), VRAM đỉnh, RAM, latency (mean/p95 ms), FPS.
Batch=1, warm-up 20 vòng, đo 200 vòng; chạy cho FP32 / FP16 / ONNX / INT8.
    python -m src.benchmark --config configs/m4_x3d.yaml --format onnx
"""
import time
import numpy as np
import torch


@torch.no_grad()
def measure_latency(model, x, warmup=20, iters=200):
    for _ in range(warmup):
        model(x)
    if x.is_cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    vram = torch.cuda.max_memory_allocated() / 2**20 if x.is_cuda else None
    return {"latency_ms": float(np.mean(ts)), "p95_ms": float(np.percentile(ts, 95)), "vram_mb": vram}
