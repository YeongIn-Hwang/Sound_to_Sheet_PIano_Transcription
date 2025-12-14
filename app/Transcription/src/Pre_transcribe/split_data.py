import os
import shutil
import random
from pathlib import Path

# =========================
# 설정 부분만 바꿔서 쓰면 됨
# =========================
DATA_DIR = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Transcription\mel"   # A 폴더 (데이터)
LABEL_DIR = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Transcription\Label" # B 폴더 (라벨 .txt)
OUTPUT_DIR = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Transcription\/2.Data"  # C 폴더 (새로 만들 루트)

TEST_RATIO = 0.1   # 테스트 비율 (10%)
VAL_RATIO = 0.1    # 검증 비율 (10%)
RANDOM_SEED = 42   # 셔플 재현용

# =========================
# 내부 함수들
# =========================

def collect_files_by_stem(directory, allowed_exts=None):
    """
    directory 아래의 모든 파일을 탐색해서
    stem(확장자 제외 이름) -> 파일 경로 로 매핑해서 반환.
    allowed_exts 가 주어지면 그 확장자만 사용.
    """
    mapping = {}
    for root, dirs, files in os.walk(directory):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if allowed_exts is not None and ext not in allowed_exts:
                continue
            stem = os.path.splitext(name)[0]
            full_path = os.path.join(root, name)
            mapping[stem] = full_path
    return mapping


def make_subset(output_root, subset_name, stems, data_files, label_files):
    """
    subset_name(train/val/test)에 해당하는 데이터/라벨 파일을
    output_root/subset_name/Data, Label 에 복사.
    """
    data_out_dir = os.path.join(output_root, subset_name, "Data")
    label_out_dir = os.path.join(output_root, subset_name, "Label")
    os.makedirs(data_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    for stem in stems:
        src_data = data_files[stem]
        src_label = label_files[stem]

        # 원래 파일명 유지
        data_dst = os.path.join(data_out_dir, os.path.basename(src_data))
        label_dst = os.path.join(label_out_dir, os.path.basename(src_label))

        shutil.move(src_data, data_dst)
        shutil.move(src_label, label_dst)

    print(f"[{subset_name}] 쌍 {len(stems)}개 이동 완료 "
          f"-> {data_out_dir}, {label_out_dir}")


def main():
    # 1. 파일 수집
    print("데이터/라벨 파일 수집 중...")

    # 데이터: 확장자 제한 없음 (A 폴더에 txt 없다고 가정)
    data_files = collect_files_by_stem(DATA_DIR, allowed_exts=None)

    # 라벨: .txt만
    label_files = collect_files_by_stem(LABEL_DIR, allowed_exts={".txt"})

    # 2. 공통으로 존재하는 stem만 사용 (데이터-라벨 쌍)
    common_stems = sorted(set(data_files.keys()) & set(label_files.keys()))
    total = len(common_stems)

    if total == 0:
        print("데이터-라벨 쌍을 하나도 찾지 못했습니다. 경로/파일명을 확인하세요.")
        return

    print(f"총 쌍 개수: {total}")

    # 3. 셔플 후 train/val/test 나누기
    random.seed(RANDOM_SEED)
    random.shuffle(common_stems)

    n_test = int(total * TEST_RATIO)
    n_val = int(total * VAL_RATIO)
    n_train = total - n_test - n_val

    if n_train <= 0:
        raise ValueError("train 개수가 0 이하입니다. 비율을 다시 설정하세요.")

    test_stems = common_stems[:n_test]
    val_stems = common_stems[n_test:n_test + n_val]
    train_stems = common_stems[n_test + n_val:]

    print(f"train: {n_train}, val: {len(val_stems)}, test: {len(test_stems)}")

    # 4. 출력 루트 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 5. 각각 복사
    make_subset(OUTPUT_DIR, "train", train_stems, data_files, label_files)
    make_subset(OUTPUT_DIR, "val",   val_stems,   data_files, label_files)
    make_subset(OUTPUT_DIR, "test",  test_stems,  data_files, label_files)

    print("\n=== 전체 완료 ===")


if __name__ == "__main__":
    main()
