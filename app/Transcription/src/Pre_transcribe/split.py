import os
import random
import shutil
from pathlib import Path

# ================================
# 경로 설정
# ================================
A_DIR = Path(r"C:\Users\hyi8402\Desktop\Sound to Sheet\Transcription\2.CQT_Data\train")
B_DIR = Path(r"C:\Users\hyi8402\Desktop\Sound to Sheet\Transcription\2.CQT_Data\val")

SRC_DATA = A_DIR / "Data"
SRC_LABEL = A_DIR / "Label"

DST_DATA = B_DIR / "Data"
DST_LABEL = B_DIR / "Label"

DST_DATA.mkdir(parents=True, exist_ok=True)
DST_LABEL.mkdir(parents=True, exist_ok=True)

# ================================
# 1. Label_split 기준으로 파일 목록 가져오기
# ================================
label_files = list(SRC_LABEL.glob("*.txt"))

print(f"총 세그먼트(label) 개수: {len(label_files)}")

# ================================
# 2. 12% 추출
# ================================
ratio = 0.12
pick_n = int(len(label_files) * ratio)

print(f"→ 이동할 세그먼트 개수: {pick_n}")

random.seed(42)  # 재현 가능하도록
picked = random.sample(label_files, pick_n)

# ================================
# 3. 이동 (라벨 + 데이터)
# ================================
for label_path in picked:
    name = label_path.stem   # 예: AkPnBcht_01_003
    data_path = SRC_DATA / f"{name}.pt"

    dst_label_path = DST_LABEL / f"{name}.txt"
    dst_data_path = DST_DATA / f"{name}.pt"

    # 라벨 이동
    if label_path.exists():
        shutil.move(str(label_path), str(dst_label_path))

    # 데이터 이동
    if data_path.exists():
        shutil.move(str(data_path), str(dst_data_path))
    else:
        print(f"[WARN] Data 파일 없음: {data_path}")

print("\n✔ 완료! 12% 세그먼트를 B 폴더로 이동했습니다.")
