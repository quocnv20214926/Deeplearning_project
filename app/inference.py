"""[Owner: TV4] Bộ đệm trượt + chạy model realtime.
- Lấy 1/10 frame (khớp frame_step lúc train), resize theo transforms.build_eval_transform (TV1).
- Đủ clip_len frame -> chạy model (ONNX Runtime/PyTorch FP16) -> (score bất thường, lớp top-1).
- Làm mượt score (moving average).
"""
