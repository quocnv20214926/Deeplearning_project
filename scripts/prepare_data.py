"""[Owner: TV1] Chuẩn bị dữ liệu.
1. Tải Kaggle:  kaggle datasets download -d odins0n/ucf-crime-dataset -p data/raw --unzip
2. Đặt file Temporal_Anomaly_Annotation_for_Testing_Videos.txt (trang UCF-Crime gốc, CRCV) vào data/annotations/
3. Script này: kiểm tra số ảnh/lớp, nhóm ảnh theo video, tạo split train/val/test -> data/processed/splits/,
   tạo data/processed/test_frame_labels.npz (nhãn 0/1 cho từng ảnh test).
"""
