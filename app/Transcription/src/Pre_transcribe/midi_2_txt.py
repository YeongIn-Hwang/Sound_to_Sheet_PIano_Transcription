import os
import pretty_midi

def midi_to_txt(midi_path: str, txt_path: str, skip_drums: bool = True):
    """개별 MIDI → TXT 변환 함수"""
    pm = pretty_midi.PrettyMIDI(midi_path)
    rows = []

    for inst in pm.instruments:
        if skip_drums and inst.is_drum:
            continue
        for note in inst.notes:
            start = float(note.start)
            end = float(note.end)
            pitch = int(note.pitch)
            rows.append((start, end, pitch))

    rows.sort(key=lambda x: (x[0], x[2], x[1]))  # onset → pitch → offset

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("OnsetTime\tOffsetTime\tMidiPitch\n")
        for start, end, pitch in rows:
            f.write(f"{start:.6f}\t{end:.6f}\t{pitch}\n")

    print(f"[OK] {os.path.basename(midi_path)} → {os.path.basename(txt_path)} "
          f"({len(rows)} notes)")


def convert_all_midis(input_dir: str, output_dir: str, skip_drums: bool = True):
    """폴더 내 모든 MIDI 파일을 TXT로 변환"""
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".mid", ".midi")):
            mid_path = os.path.join(input_dir, filename)
            txt_name = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(output_dir, txt_name)

            midi_to_txt(mid_path, txt_path, skip_drums)
            count += 1

    print(f"\n변환 완료! 총 {count}개의 MIDI 파일을 처리했습니다.")


# --------------------------
# ★ 사용법 예시 ★
# --------------------------
if __name__ == "__main__":
    INPUT_DIR = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Transcription\1.Pre\Maestro"   # MIDI 폴더 경로
    OUTPUT_DIR = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Transcription\2.CQT_Data\train\Label"   # TXT 저장 폴더 경로

    convert_all_midis(INPUT_DIR, OUTPUT_DIR)
