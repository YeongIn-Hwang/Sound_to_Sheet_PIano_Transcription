import os
import torchaudio
import torch
import soundfile as sf   # ✅ 추가
from pathlib import Path

# =============================
# 설정
# =============================
INPUT_DIR = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Train_sound\Maestro"   # A 폴더
OUTPUT_DIR = r"C:\Users\hyi8402\Desktop\Sound to Sheet\mel"  # B 폴더 (없으면 자동 생성)
SR = 16000

# =============================
# Mel 변환기 정의 (Onsets & Frames 기준)
# =============================
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR,
    n_fft=2048,
    hop_length=512,
    win_length=2048,
    window_fn=torch.hann_window,
    n_mels=229,
    f_min=30.0,
    f_max=8000.0,
    power=2.0,
)

def wav_to_mel(path):
    # ⚠️ Path 객체일 수 있으니 문자열로 변환
    path = str(path)

    # 1) soundfile로 오디오 로드 (numpy array 반환)
    audio, orig_sr = sf.read(path)  # shape: (N,) or (N, C)

    # 2) numpy -> torch tensor
    audio = torch.as_tensor(audio, dtype=torch.float32)

    # 3) Stereo -> Mono
    if audio.ndim == 2:          # (N, C)
        audio = audio.mean(dim=1)  # (N,)

    # 4) (1, N) 형태로 reshape
    wav = audio.unsqueeze(0)     # (1, N)

    # 5) Resample (orig_sr -> SR)
    if orig_sr != SR:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=SR)
        wav = resampler(wav)

    # 6) Mel transform
    mel = mel_transform(wav)

    # 7) Log scaling
    mel = torch.log(mel + 1e-6)

    return mel


# =============================
# 전체 처리 함수 (이 부분은 그대로)
# =============================
def process_all_wavs(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = list(input_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files.")

    for wav_path in wav_files:
        rel_path = wav_path.relative_to(input_dir)
        save_dir = output_dir / rel_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)

        mel = wav_to_mel(wav_path)

        save_path = save_dir / (wav_path.stem + ".pt")
        torch.save(mel, save_path)

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    process_all_wavs(INPUT_DIR, OUTPUT_DIR)
