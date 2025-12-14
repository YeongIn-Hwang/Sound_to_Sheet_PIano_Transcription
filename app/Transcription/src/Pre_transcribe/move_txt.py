import os
import shutil

def get_non_duplicate_path(dst_dir, filename):
    """
    중복되지 않는 새로운 파일 경로 생성
    example.txt -> example (2).txt -> example (3).txt ...
    """
    base, ext = os.path.splitext(filename)
    counter = 2

    new_filename = filename
    new_path = os.path.join(dst_dir, new_filename)

    while os.path.exists(new_path):
        new_filename = f"{base} ({counter}){ext}"
        new_path = os.path.join(dst_dir, new_filename)
        counter += 1

    return new_path


def move_all_txts(src_dir: str, dst_dir: str):
    """
    src_dir 및 모든 하위 폴더에서 .txt 파일을 찾아
    dst_dir로 이동한다.

    - 같은 이름의 파일이 이미 있으면 (2), (3) 등 번호 붙여 저장
    - 덮어쓰기 없음
    """

    os.makedirs(dst_dir, exist_ok=True)

    moved = 0
    renamed = 0

    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(".txt"):
                src_path = os.path.join(root, f)

                # 중복 처리
                dst_path = get_non_duplicate_path(dst_dir, f)
                if os.path.basename(dst_path) != f:
                    renamed += 1

                shutil.move(src_path, dst_path)
                moved += 1
                print(f"[MOVE] {src_path} -> {dst_path}")

    print("\n=== TXT 이동 완료 ===")
    print(f"총 이동된 TXT: {moved}")
    print(f"이름 바뀐 파일: {renamed}")


# 사용 예시
if __name__ == "__main__":
    SRC_DIR = r"C:\Users\hyi8402\Desktop\Sound to Sheet"       # TXT들이 흩어져 있는 루트 폴더
    DST_DIR = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Label"    # TXT 모을 폴더

    move_all_txts(SRC_DIR, DST_DIR)
