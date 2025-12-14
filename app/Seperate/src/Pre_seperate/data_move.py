import os
import shutil
import glob
import yaml  # pip install pyyaml

# =========================================
# 설정
# =========================================
SRC_ROOT = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Dataset\slakh2100\validation"  # A 폴더 (원본)
DST_ROOT = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Dataset\slakh2100\val_piano" # B 폴더 (결과 저장 위치)


def load_metadata(meta_path):
    """metadata.yaml 로드"""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    return meta


def find_piano_stems(meta):
    """metadata.yaml에서 inst_class == 'Piano' 인 stem ID 리스트 반환 (예: ['S00', 'S03'])"""
    piano_stems = []
    stems = meta.get("stems", {})
    for stem_id, info in stems.items():
        if info.get("inst_class") == "Piano":
            piano_stems.append(stem_id)
    return piano_stems


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def copy_if_exists(src, dst_dir):
    """src 파일이 있으면 dst_dir로 복사 (파일명 유지)"""
    if os.path.isfile(src):
        ensure_dir(dst_dir)
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
        return True
    return False


def process_track(track_dir, dst_root):
    """
    하나의 Track 폴더를 보고:
      - Piano stem이 있으면
      - mix.flac 복사
      - Piano stem audio + MIDI 복사
    """
    track_name = os.path.basename(track_dir)
    meta_path = os.path.join(track_dir, "metadata.yaml")
    if not os.path.isfile(meta_path):
        print(f"[SKIP] {track_name}: metadata.yaml 없음")
        return

    meta = load_metadata(meta_path)
    piano_stems = find_piano_stems(meta)

    if not piano_stems:
        print(f"[SKIP] {track_name}: Piano stem 없음")
        return

    print(f"[TRACK] {track_name}: Piano stems = {piano_stems}")

    # 목적지 Track 폴더
    dst_track_dir = os.path.join(dst_root, track_name)
    ensure_dir(dst_track_dir)

    # ------------------------------------------------
    # 1) mix.flac (또는 mix.wav) 복사
    # ------------------------------------------------
    mix_cands = [
        os.path.join(track_dir, "mix.flac"),
        os.path.join(track_dir, "mix.wav"),
    ]
    copied_mix = False
    for cand in mix_cands:
        if os.path.isfile(cand):
            shutil.copy2(cand, os.path.join(dst_track_dir, os.path.basename(cand)))
            copied_mix = True
            break
    if not copied_mix:
        print(f"  [WARN] {track_name}: mix.flac / mix.wav 없음")

    # ------------------------------------------------
    # 2) stems 안에서 piano stem 파일 복사
    # ------------------------------------------------
    src_stems_dir = os.path.join(track_dir, "stems")
    dst_stems_dir = os.path.join(dst_track_dir, "stems")
    if not os.path.isdir(src_stems_dir):
        print(f"  [WARN] {track_name}: stems 디렉토리 없음")
    else:
        for stem_id in piano_stems:
            # S00.*, S00_*.flac 등 다양한 확장자/이름 패턴 대비
            pattern = os.path.join(src_stems_dir, f"{stem_id}*")
            matches = glob.glob(pattern)
            if not matches:
                print(f"  [WARN] {track_name}: stems/{stem_id}* 파일 없음")
                continue
            ensure_dir(dst_stems_dir)
            for src_f in matches:
                shutil.copy2(src_f, os.path.join(dst_stems_dir, os.path.basename(src_f)))
                print(f"    [COPY] stems: {os.path.basename(src_f)}")

    # ------------------------------------------------
    # 3) MIDI 안에서 piano stem MIDI 파일 복사
    # ------------------------------------------------
    src_midi_dir = os.path.join(track_dir, "MIDI")
    dst_midi_dir = os.path.join(dst_track_dir, "MIDI")
    if not os.path.isdir(src_midi_dir):
        print(f"  [WARN] {track_name}: MIDI 디렉토리 없음")
    else:
        for stem_id in piano_stems:
            # S00.mid, S00_*.mid 등 대응
            pattern_mid = os.path.join(src_midi_dir, f"{stem_id}*.mid*")
            matches_mid = glob.glob(pattern_mid)
            if not matches_mid:
                print(f"  [WARN] {track_name}: MIDI/{stem_id}*.mid* 파일 없음")
                continue
            ensure_dir(dst_midi_dir)
            for src_f in matches_mid:
                shutil.copy2(src_f, os.path.join(dst_midi_dir, os.path.basename(src_f)))
                print(f"    [COPY] MIDI: {os.path.basename(src_f)}")

    # (선택) metadata.yaml도 같이 복사하고 싶으면:
    shutil.copy2(meta_path, os.path.join(dst_track_dir, "metadata.yaml"))


def main():
    # SRC_ROOT 아래에서 "Track"으로 시작하는 폴더들만 순회
    for name in os.listdir(SRC_ROOT):
        track_dir = os.path.join(SRC_ROOT, name)
        if not os.path.isdir(track_dir):
            continue
        if not name.lower().startswith("track"):
            continue

        process_track(track_dir, DST_ROOT)


if __name__ == "__main__":
    main()
