"""[Owner: TV4] Export backbone+head -> ONNX (FP32/FP16), quantize INT8 (onnxruntime.quantization).
    python -m src.export_onnx --config configs/m4_x3d.yaml --ckpt weights/x3d_s_cls.pt
"""
