"""[Owner: TV2] Đánh giá trên 290 video test:
frame AUC, AUC theo lớp, FPR@0.5, FPR@TPR90, accuracy 14 lớp, ROC curve.
Kết quả ghi results/tables/<model>_<mode>.json, điểm từng frame -> results/predictions/.
    python -m src.evaluate --config configs/m1_r2plus1d.yaml --ckpt weights/r2plus1d_18_cls.pt --mode cls
"""
