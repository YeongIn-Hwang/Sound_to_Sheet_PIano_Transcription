import os
import shutil

def move_all_npz_to_B(root_dir, B_dir):
    """
    root_dir: 탐색 시작할 최상위 폴더 경로
    B_dir: npz 파일을 모두 모아놓을 폴더 경로
    """

    # B 폴더 없으면 생성
    os.makedirs(B_dir, exist_ok=True)

    # 재귀적으로 모든 파일 탐색
    for current_path, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".npz"):
                src_path = os.path.join(current_path, file)
                dst_path = os.path.join(B_dir, file)

                # 파일명이 이미 존재하는 경우 → 뒤에 숫자 붙이기
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(file)
                    count = 1
                    new_name = f"{base}_{count}{ext}"
                    dst_path = os.path.join(B_dir, new_name)

                    while os.path.exists(dst_path):
                        count += 1
                        new_name = f"{base}_{count}{ext}"
                        dst_path = os.path.join(B_dir, new_name)

                print(f"Moving: {src_path} → {dst_path}")
                shutil.move(src_path, dst_path)

    print("\n=== 완료: 모든 .npz 파일이 B 폴더로 이동되었습니다. ===")

A = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Dataset\lpd_5\lpd_5_cleansed"
B = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Dataset\lpd_5\npz"

# 사용 예시
move_all_npz_to_B(A,B)
