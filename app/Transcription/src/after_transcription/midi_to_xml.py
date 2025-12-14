from music21 import converter
from pathlib import Path

# =========================
# 하드코딩된 MIDI 경로
# =========================
MIDI_PATH = Path(
    r"C:\Users\hyi8402\Desktop\CQT\홍순_hrplus.mid"
)

# =========================
# 출력 XML 경로
# =========================
OUTPUT_XML_PATH = MIDI_PATH.with_suffix(".musicxml")

def midi_to_musicxml(midi_path: Path, output_path: Path):
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI 파일이 존재하지 않습니다: {midi_path}")

    print(f"[INFO] MIDI 로드 중: {midi_path}")

    score = converter.parse(str(midi_path))

    print(f"[INFO] MusicXML 생성 중...")
    score.write("musicxml", str(output_path))

    print(f"[DONE] MusicXML 저장 완료:")
    print(f"       {output_path}")


if __name__ == "__main__":
    midi_to_musicxml(MIDI_PATH, OUTPUT_XML_PATH)
