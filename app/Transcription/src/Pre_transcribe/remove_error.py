import os
import re

def remove_number_suffix(directory):
    """
    directory 아래의 모든 파일에서
    ' (숫자)' 패턴 제거
    
    예) 'track (2).txt' -> 'track.txt'
    """

    pattern = re.compile(r"(.*?) \(\d+\)(\.[a-zA-Z0-9]+)$")

    for root, dirs, files in os.walk(directory):
        for f in files:
            match = pattern.match(f)

            if match:
                # 원래 파일명 복구
                original_name = match.group(1) + match.group(2)
                
                old_path = os.path.join(root, f)
                new_path = os.path.join(root, original_name)

                # 같은 이름이 이미 있으면 덮어쓰지 않음
                if os.path.exists(new_path):
                    print(f"[SKIP] {new_path} already exists")
                    continue

                os.rename(old_path, new_path)
                print(f"[RENAME] {f} → {original_name}")


# 사용 예시
if __name__ == "__main__":
    TARGET_DIR = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Label"
    remove_number_suffix(TARGET_DIR)
