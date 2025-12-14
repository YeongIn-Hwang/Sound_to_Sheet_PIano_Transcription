# infer.py
import math
from pathlib import Path

import torch
import torchaudio

from model import HTDemucs

# =========================
# 설정 (train.py와 동일하게)
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = r"C:\Users\hyi8402\Desktop\Sound to Sheet\app\Seperate\src\htde\best.pth"
INPUT_AUDIO = r"C:\Users\hyi8402\Desktop\CQT\riding.mp3"
OUT_DIR = Path(r"C:\Users\hyi8402\Desktop\infer_out")

TARGET_SR = 22050       # train.py의 TARGET_SR
SEGMENT_SEC = 6.0       # train.py의 SEGMENT_SECONDS

# 오버랩(초) - 너무 작으면 경계 뚝 끊김, 너무 크면 느려짐
OVERLAP_SEC = 1.0       # 0.5~1.5 사이 추천


def load_audio_any(path: str, target_sr: int) -> torch.Tensor:
    """
    어떤 포맷이든 torchaudio로 로드.
    반환: (C, T) float32
    """
    wav, sr = torchaudio.load(path)  # (C, T)
    wav = wav.to(torch.float32)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def to_mono(wav: torch.Tensor) -> torch.Tensor:
    """
    (C, T) -> (T) mono
    """
    if wav.dim() != 2:
        raise ValueError(f"wav must be (C,T). got {wav.shape}")
    if wav.size(0) == 1:
        return wav[0]
    return wav.mean(dim=0)


def save_wav(path: Path, wav_1d: torch.Tensor, sr: int):
    """
    wav_1d: (T) or (1,T)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if wav_1d.dim() == 1:
        wav_1d = wav_1d.unsqueeze(0)  # (1,T)
    torchaudio.save(str(path), wav_1d.cpu(), sr)


def make_fade_window(length: int, device: torch.device) -> torch.Tensor:
    """
    overlap-add용 윈도우 (Hann)
    """
    if length <= 1:
        return torch.ones(length, device=device)
    return torch.hann_window(length, device=device, periodic=False)


@torch.no_grad()
def separate_long_audio(model: torch.nn.Module, mono_wav: torch.Tensor) -> torch.Tensor:
    """
    mono_wav: (T) on CPU or GPU (상관없음)
    반환: pred_mono (T)  (piano)
    """
    model.eval()

    seg_len = int(SEGMENT_SEC * TARGET_SR)
    hop_len = int((SEGMENT_SEC - OVERLAP_SEC) * TARGET_SR)
    hop_len = max(1, hop_len)

    T = mono_wav.numel()
    if T <= seg_len:
        # (B,1,T)
        x = mono_wav.unsqueeze(0).unsqueeze(0).to(DEVICE)
        y = model(x)               # (B,S=1,C=1,T)
        return y[0, 0, 0].detach().cpu()

    # overlap-add
    out = torch.zeros(T, device="cpu")
    wsum = torch.zeros(T, device="cpu")

    fade = make_fade_window(seg_len, torch.device("cpu"))

    n_chunks = math.ceil((T - seg_len) / hop_len) + 1
    for i in range(n_chunks):
        s = i * hop_len
        e = s + seg_len

        chunk = mono_wav[s:e]
        if chunk.numel() < seg_len:
            pad = seg_len - chunk.numel()
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        x = chunk.unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,seg_len)
        y = model(x)                                    # (1,1,1,seg_len)
        pred = y[0, 0, 0].detach().cpu()                 # (seg_len)

        # 원래 길이만큼만 overlap-add
        valid = min(seg_len, T - s)
        out[s:s+valid] += pred[:valid] * fade[:valid]
        wsum[s:s+valid] += fade[:valid]

    out = out / (wsum.clamp_min(1e-8))
    return out


@torch.no_grad()
def main():
    print(f"[Info] DEVICE: {DEVICE}")

    # ✅ train.py와 동일한 모델 형태로 생성
    model = HTDemucs(sources=["piano"], audio_channels=1).to(DEVICE)

    # ✅ train.py 저장 포맷: state["model"]에 진짜 state_dict가 들어있음
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        # 혹시 state_dict만 저장된 파일일 경우 대비
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print("[Warn] Missing keys (일부 누락):", missing[:10], f"... (+{len(missing)-10})" if len(missing) > 10 else "")
    if unexpected:
        print("[Warn] Unexpected keys (불필요):", unexpected[:10], f"... (+{len(unexpected)-10})" if len(unexpected) > 10 else "")

    # 오디오 로드 -> mono
    wav = load_audio_any(INPUT_AUDIO, TARGET_SR)  # (C,T)
    mono = to_mono(wav)                           # (T)

    # 추론 (긴 오디오 청크 처리)
    pred = separate_long_audio(model, mono)       # (T) on CPU

    # 저장
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "piano.wav"
    save_wav(out_path, pred, TARGET_SR)

    print(f"✅ Done: {out_path}")


if __name__ == "__main__":
    main()
