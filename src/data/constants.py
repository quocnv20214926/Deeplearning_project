"""[Owner: TV1] Hằng số dữ liệu UCF-Crime dùng chung."""

NORMAL = "NormalVideos"
# class_id 0 = Normal, nên nhãn nhị phân bất thường = (class_id > 0)
CLASSES = [
    NORMAL, "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
    "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

SOURCE_SIZE = 64  # bản Kaggle: ảnh 64x64
