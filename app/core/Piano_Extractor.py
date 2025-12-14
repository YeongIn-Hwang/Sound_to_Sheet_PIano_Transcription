from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio

# infer.py가 from model import HTDemucs 를 쓰고 있으니, 네 프로젝트 구조에 맞게 수정 :contentReference[oaicite:1]{index=1}
# 예: from Seperate.src.htde.model import HTDemucs  (실제 파일 위치에 맞춰)

from Seperate.src.htde.model import HTDemucs


def load_audio_any(path: str, target_sr: int) -> torch.Tensor:
    """
    어떤 포맷이든 torchaudio로 로드.
    반환: (C, T) float32
    """
    wav, sr = torchaudio.load(path)  # (C,T)
    wav = wav.to(torch.float32)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def to_mono(wav: torch.Tensor) -> torch.Tensor:
    """
    (C, T) -> (T)
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
def separate_long_audio(
    model: torch.nn.Module,
    mono_wav: torch.Tensor,
    device: torch.device,
    target_sr: int,
    segment_sec: float,
    overlap_sec: float,
) -> torch.Tensor:
    """
    mono_wav: (T)  (CPU/GPU 상관없음)
    반환: pred_mono (T) (piano)
    """
    model.eval()

    seg_len = int(segment_sec * target_sr)
    hop_len = int((segment_sec - overlap_sec) * target_sr)
    hop_len = max(1, hop_len)

    T = mono_wav.numel()
    if T <= seg_len:
        x = mono_wav.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T)
        y = model(x)  # (B,S=1,C=1,T)
        return y[0, 0, 0].detach().cpu()

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
            chunk = F.pad(chunk, (0, pad))

        x = chunk.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,seg_len)
        y = model(x)                                    # (1,1,1,seg_len)
        pred = y[0, 0, 0].detach().cpu()                # (seg_len)

        valid = min(seg_len, T - s)
        out[s:s+valid] += pred[:valid] * fade[:valid]
        wsum[s:s+valid] += fade[:valid]

    out = out / (wsum.clamp_min(1e-8))
    return out


@dataclass
class PianoExtractorConfig:
    ckpt_path: str
    target_sr: int = 22050
    segment_sec: float = 6.0
    overlap_sec: float = 1.0
    device: Optional[str] = None  # "cuda" | "cpu" | None(auto)


class PianoExtractor:
    """
    HTDemucs 기반 피아노 추출기.
    - 서버 시작 시 1회 모델 로드
    - extract_piano_to_wav: 파일로 저장해서 FastAPI가 FileResponse/bytes로 내려주기 쉬움
    """

    def __init__(self, cfg: PianoExtractorConfig):
        self.ckpt_path = str(cfg.ckpt_path)
        self.target_sr = int(cfg.target_sr)
        self.segment_sec = float(cfg.segment_sec)
        self.overlap_sec = float(cfg.overlap_sec)

        if cfg.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        # 모델 생성 (infer.py와 동일: sources=["piano"], audio_channels=1) :contentReference[oaicite:2]{index=2}
        self.model = HTDemucs(sources=["piano"], audio_channels=1).to(self.device)

        # 체크포인트 로드 (infer.py 로직 그대로: dict면 "model" 키 확인) :contentReference[oaicite:3]{index=3}
        ckpt = torch.load(
            self.ckpt_path,
            map_location="cpu",
            weights_only=True,
        )
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt

        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def extract_piano_to_wav(self, input_audio_path: str, output_wav_path: str) -> str:
        """
        input_audio_path: mp3/wav/flac 경로
        output_wav_path: 저장할 wav 경로
        """
        wav = load_audio_any(input_audio_path, self.target_sr)  # (C,T)
        mono = to_mono(wav)                                     # (T)

        pred = separate_long_audio(
            model=self.model,
            mono_wav=mono,
            device=self.device,
            target_sr=self.target_sr,
            segment_sec=self.segment_sec,
            overlap_sec=self.overlap_sec,
        )  # (T) on CPU

        out_path = Path(output_wav_path)
        save_wav(out_path, pred, self.target_sr)
        return str(out_path)
