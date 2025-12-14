# make_cqt_dataset.py

import os
from pathlib import Path

import librosa
import numpy as np
import torch

# ==========================
# 하드코딩 설정
# ==========================
# 원본 오디오 루트 (여기만 너 환경에 맞게 바꾸면 됨)
INPUT_ROOT  = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Transcription\1.Pre\Maestro"
# CQT 저장할 루트
OUTPUT_ROOT = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Transcription\2.CQT_Data\train\Data"

SR   = 16000
HOP  = 512           # FRAME_TIME = 0.032s 과 동일하게
N_BINS = 88          # 피아노 88키
FMIN  = librosa.note_to_hz("A0")  # 27.5 Hz

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


def audio_to_cqt(path: Path) -> torch.Tensor:
    """
    단일 오디오 파일(path)을 읽어서
    (1, n_bins, T) 형태의 CQT magnitude 텐서를 반환.
    """
    y, sr = librosa.load(path.as_posix(), sr=SR, mono=True)
    # CQT 계산 (복소값)
    cqt = librosa.cqt(
        y,
        sr=SR,
        hop_length=HOP,
        fmin=FMIN,
        n_bins=N_BINS,
        bins_per_octave=12,
    )
    # magnitude로 변환
    cqt_mag = np.abs(cqt).astype(np.float32)      # (n_bins, T)
    cqt_tensor = torch.from_numpy(cqt_mag)        # (n_bins, T)

    # Data_load.MelCache와 맞추기 위해 (1, n_bins, T) 형태로
    if cqt_tensor.dim() == 2:
        cqt_tensor = cqt_tensor.unsqueeze(0)

    return cqt_tensor


def process_all_audio(input_root: str, output_root: str):
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    count = 0
    for dirpath, dirnames, filenames in os.walk(input_root):
        dirpath = Path(dirpath)
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext not in AUDIO_EXTS:
                continue

            in_path = dirpath / fname

            # stem: 확장자 제거한 파일 이름만 사용
            # 예: AkPnBcht_01.wav → AkPnBcht_01.pt
            stem = in_path.stem
            out_path = output_root / f"{stem}.pt"

            if out_path.exists():
                print(f"[SKIP] {out_path} already exists")
                continue

            print(f"[CQT] {in_path} -> {out_path}")
            try:
                cqt_tensor = audio_to_cqt(in_path)
                torch.save(cqt_tensor, out_path)
                count += 1
            except Exception as e:
                print(f"[ERROR] {in_path}: {e}")

    print(f"\nDone. Created CQT for {count} files.")


if __name__ == "__main__":
    process_all_audio(INPUT_ROOT, OUTPUT_ROOT)
