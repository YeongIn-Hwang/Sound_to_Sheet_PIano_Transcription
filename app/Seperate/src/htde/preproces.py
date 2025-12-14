# preprocess_slakh_piano.py

import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

TARGET_SR = 22050


def load_audio_mono(path: Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    """
    path: flac 파일 경로
    return: (T,) float32, [-1,1] 근처
    """
    wav, sr = torchaudio.load(str(path))  # (C, T)
    # resample
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    # mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # (T,)
    wav = wav.squeeze(0).to(torch.float32)
    return wav


def align_length(mix: torch.Tensor, piano: torch.Tensor):
    T = min(mix.shape[-1], piano.shape[-1])
    return mix[..., :T], piano[..., :T]


def normalize_pair(mix: torch.Tensor, piano: torch.Tensor, eps: float = 1e-6):
    peak = max(mix.abs().max().item(), piano.abs().max().item(), eps)
    mix = mix / peak
    piano = piano / peak
    return mix, piano


def process_split(root_dir: Path, split: str = "train"):
    """
    root_dir/split 안의 각 Track 폴더를 순회하며 mix_piano.pt 생성
    """
    split_dir = root_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"{split_dir} not found.")

    track_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
    print(f"[{split}] Found {len(track_dirs)} track dirs")

    for track_dir in tqdm(track_dirs, desc=f"Preprocessing {split}"):
        out_pt = track_dir / "mix_piano.pt"
        if out_pt.exists():
            # 이미 처리된 경우 스킵하고 싶으면 여기서 continue
            # continue
            pass

        mix_path = track_dir / "mix.flac"
        piano_path = track_dir / "piano_mix.flac"

        if not mix_path.exists() or not piano_path.exists():
            print(f"  [WARN] {track_dir} missing mix.flac or piano_mix.flac, skip")
            continue

        try:
            mix = load_audio_mono(mix_path, TARGET_SR)
            piano = load_audio_mono(piano_path, TARGET_SR)
        except Exception as e:
            print(f"  [ERROR] loading audio in {track_dir}: {e}")
            continue

        mix, piano = align_length(mix, piano)
        mix, piano = normalize_pair(mix, piano)

        data = {
            "mix": mix,       # (T,)
            "piano": piano,   # (T,)
            "sr": TARGET_SR,
        }
        torch.save(data, str(out_pt))


def main():
    # 여기 root_dir만 네 Slakh 전처리 폴더로 맞춰주면 됨
    root_dir = Path(r"C:\Users\hyi8402\Desktop\Sound to Sheet\Dataset\slakh2100")  # 예시

    for split in ["train_piano", "val_piano", "test_piano"]:
        if (root_dir / split).exists():
            process_split(root_dir, split)
        else:
            print(f"[INFO] Split '{split}' not found, skip.")


if __name__ == "__main__":
    main()
