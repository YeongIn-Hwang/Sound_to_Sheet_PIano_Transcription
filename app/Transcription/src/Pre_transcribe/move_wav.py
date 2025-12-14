import os
import shutil

def get_non_duplicate_path(dst_dir, filename):
    """
    중복되지 않는 새로운 파일 경로 생성
    example.wav → example (2).wav → example (3).wav …
    """
    base, ext = os.path.splitext(filename)
    counter = 2

    new_filename = filename
    new_path = os.path.join(dst_dir, new_filename)

    # 파일이 존재하면 번호 증가시키며 새로운 이름 생성
    while os.path.exists(new_path):
        new_filename = f"{base} ({counter}){ext}"
        new_path = os.path.join(dst_dir, new_filename)
        counter += 1

    return new_path


def move_all_wavs(src_dir: str, dst_dir: str):
    """
    src_dir 및 하위 모든 폴더에서 .wav 파일을 찾아
    dst_dir로 이동한다.

    - 중복 파일명 존재 시 자동으로 (2), (3) 등 붙여 저장
    - 덮어쓰기 없음
    """

    os.makedirs(dst_dir, exist_ok=True)

    moved = 0
    renamed = 0

    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                src_path = os.path.join(root, f)

                # 중복 체크 → 새로운 파일명 생성
                dst_path = get_non_duplicate_path(dst_dir, f)
                if os.path.basename(dst_path) != f:
                    renamed += 1

                # 이동
                shutil.move(src_path, dst_path)
                moved += 1
                print(f"[MOVE] {src_path} → {dst_path}")

    print("\n=== 완료 ===")
    print(f"총 이동된 WAV: {moved}")
    print(f"중복으로 새 이름 생성된 파일: {renamed}")


# ★ 사용 예시
if __name__ == "__main__":
    SRC_DIR = r"C:\Users\hyi8402\Desktop\Sound to Sheet\maestro-v3.0.0"
    DST_DIR = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Train_sound\Maestro"

    move_all_wavs(SRC_DIR, DST_DIR)
