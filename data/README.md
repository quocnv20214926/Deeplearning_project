# Dữ liệu (không commit lên git)
```
data/
├── raw/ucf_crime/{Train,Test}/<Class>/<Video>_<idx>.png   # Kaggle odins0n (64x64, 1/10 frame)
├── annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt
├── processed/splits/{train,val,test}.txt, test_frame_labels.npz
├── features/<model>/<split>/<video>.npy                      # cho chế độ MIL
└── external/rwf2000/                                         # cross-dataset (TV2)
```
