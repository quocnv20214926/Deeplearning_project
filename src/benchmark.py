"""[Owner: TV2] Benchmark: params, kích thước file, VRAM đỉnh, RAM, latency mean/p95, FPS.
Quy ước đo: batch=1, 1 clip (3 x clip_len x input_size x input_size), warm-up rồi đo `iters` lần.
Mỗi cấu hình nên chạy trong 1 process riêng (để RAM/VRAM không bị nhiễm từ lần chạy trước).

    python -m src.benchmark --config configs/m1_r2plus1d.yaml --precision fp32          # không cần ckpt
    python -m src.benchmark --config configs/m1_r2plus1d.yaml --ckpt weights/r2plus1d_18_cls.pt --precision fp32
    python -m src.benchmark --config ... --mode cls --precision fp16 --device cuda
    python -m src.benchmark --config ... --mode cls --precision onnx --onnx weights/r2plus1d_18_cls.onnx

FPS = clip_len x clips/giây (số frame của clip được xử lý mỗi giây, trong không gian frame đã lấy mẫu).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from src.utils.config import load_config


def summarize_latencies(times_ms, clip_len=16):
    t = np.asarray(times_ms, dtype=float)
    mean = float(t.mean())
    return {
        "latency_ms_mean": mean,
        "latency_ms_p95": float(np.percentile(t, 95)),
        "latency_ms_std": float(t.std()),
        "clips_per_s": 1000.0 / mean,
        "fps": clip_len * 1000.0 / mean,
    }


def ram_mb():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def bench_torch(model, x, device, warmup, iters):
    import torch

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync()
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            sync()
            times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_onnx(path, shape, warmup, iters, device):
    import onnxruntime as ort

    prov = ["CPUExecutionProvider"]
    if device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
        prov.insert(0, "CUDAExecutionProvider")
    sess = ort.InferenceSession(str(path), providers=prov)
    inp = sess.get_inputs()[0]
    dtype = np.float16 if "float16" in inp.type else np.float32
    x = np.random.randn(*shape).astype(dtype)
    for _ in range(warmup):
        sess.run(None, {inp.name: x})
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        times.append((time.perf_counter() - t0) * 1000)
    return times, sess.get_providers()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KHOA=GIA_TRI")
    ap.add_argument("--ckpt", help="tuỳ chọn: tốc độ/VRAM không phụ thuộc trọng số; có ckpt thì báo thêm kích thước file")
    ap.add_argument("--mode", choices=["cls", "mil"], default="cls")
    ap.add_argument("--precision", choices=["fp32", "fp16", "onnx"], default="fp32")
    ap.add_argument("--onnx", help="đường dẫn .onnx khi --precision onnx")
    ap.add_argument("--device")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--tag")
    ap.add_argument("--out_dir", default="results/tables")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    d = cfg["data"]
    shape = (1, 3, d["clip_len"], d["input_size"], d["input_size"])
    tag = args.tag or Path(args.config).stem
    rec = {"model": tag, "mode": args.mode, "precision": args.precision, "batch": 1,
           "input_shape": list(shape), "warmup": args.warmup, "iters": args.iters}

    if args.precision == "onnx":
        if not args.onnx:
            ap.error("--onnx là bắt buộc khi --precision onnx")
        device = args.device or "cpu"
        times, provider = bench_onnx(args.onnx, shape, args.warmup, args.iters, device)
        rec.update({"device": provider, "file_mb": Path(args.onnx).stat().st_size / 2**20,
                    "ram_mb": ram_mb()})
    else:
        import torch

        from src.utils.model_io import build_video_model, load_model   # TV3

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if args.precision == "fp16" and device == "cpu":
            raise SystemExit("fp16 cần GPU (--device cuda)")
        if args.ckpt:
            model = load_model(cfg, args.mode, args.ckpt, device)
        else:  # tốc độ/VRAM không phụ thuộc giá trị trọng số
            model = build_video_model(cfg, args.mode, pretrained=False)[0].to(device).eval()
        x = torch.randn(*shape, device=device)
        if args.precision == "fp16":
            model, x = model.half(), x.half()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            rec["gpu"] = torch.cuda.get_device_name()
        times = bench_torch(model, x, device, args.warmup, args.iters)
        rec.update({
            "device": device,
            "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
            "file_mb": Path(args.ckpt).stat().st_size / 2**20 if args.ckpt else None,
            "vram_peak_mb": (torch.cuda.max_memory_allocated() / 2**20
                             if device.startswith("cuda") else None),
            "ram_mb": ram_mb(),
        })
    rec.update(summarize_latencies(times, d["clip_len"]))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}_{args.mode}_{args.precision}_bench.json"
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"{tag}/{args.mode}/{args.precision}: {rec['latency_ms_mean']:.1f} ms "
          f"(p95 {rec['latency_ms_p95']:.1f}), {rec['fps']:.0f} FPS -> {path}")


if __name__ == "__main__":
    main()
