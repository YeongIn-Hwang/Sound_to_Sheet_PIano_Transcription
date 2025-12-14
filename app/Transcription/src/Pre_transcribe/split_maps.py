import os
import shutil
from pathlib import Path

# === 경로 설정 ===
SRC = Path(r"C:\Users\hyi8402\Desktop\Sound to Sheet\Transcription\1.Pre\Train_sound")  # 혼합 폴더
MAESTRO_DST = Path(r"C:\Users\hyi8402\Desktop\Sound to Sheet\Transcription\1.Pre\Maestro")
MAPS_DST = Path(r"C:\Users\hyi8402\Desktop\Sound to Sheet\Transcription\1.Pre\MAPS")

MAESTRO_DST.mkdir(parents=True, exist_ok=True)
MAPS_DST.mkdir(parents=True, exist_ok=True)


def move_pair(audio_path: Path, midi_path: Path, dest_dir: Path):
    print(f"  → 복사: {audio_path.name} + {midi_path.name}  →  {dest_dir}")
    shutil.copy2(audio_path, dest_dir / audio_path.name)
    shutil.copy2(midi_path, dest_dir / midi_path.name)


audio_exts = [".wav", ".flac"]

total_files = 0
total_audio = 0
matched_pairs = 0
no_midi = 0
maps_count = 0
maestro_count = 0

print(f"🔍 SRC 폴더 스캔 시작: {SRC}")
print("-" * 60)

for entry in SRC.iterdir():
    total_files += 1
    print(f"\n[파일 탐색] {entry.name}")

    # 오디오만 처리
    if entry.suffix.lower() not in audio_exts:
        print(f"  └ 오디오 아님({entry.suffix}) → 스킵")
        continue

    audio = entry
    total_audio += 1
    stem = audio.stem
    print(f"  ✔ 오디오 파일로 인식: {audio.name} (stem='{stem}')")

    midi = None

    # MIDI 찾기
    print("  → MIDI 후보 탐색:")
    for ext in [".mid", ".midi"]:
        candidate = SRC / f"{stem}{ext}"
        print(f"     - 찾는 중: {candidate.name} ... ", end="")
        if candidate.exists():
            midi = candidate
            print("✅ 발견!")
            break
        else:
            print("없음")

    if midi is None:
        print(f"  ⚠ MIDI 없음: {audio.name} → 이 오디오는 무시됨")
        no_midi += 1
        continue

    # 분류 기준: stem 앞 4글자
    prefix4 = stem[:4].upper()
    print(f"  → stem[:4] = '{prefix4}'")

    if prefix4 == "MAPS":
        print("  → 분류: MAPS 데이터로 인식")
        move_pair(audio, midi, MAPS_DST)
        maps_count += 1
    else:
        print("  → 분류: MAESTRO 데이터로 인식")
        move_pair(audio, midi, MAESTRO_DST)
        maestro_count += 1

    matched_pairs += 1

print("\n" + "=" * 60)
print("✅ 분류 작업 완료")
print(f"총 파일 수:            {total_files}")
print(f"그 중 오디오 파일 수:  {total_audio}")
print(f"매칭된 (audio+midi) 쌍: {matched_pairs}")
print(f"  ├ MAPS  : {maps_count}")
print(f"  └ MAESTRO: {maestro_count}")
print(f"MIDI를 찾지 못한 오디오: {no_midi}")
print("=" * 60)
