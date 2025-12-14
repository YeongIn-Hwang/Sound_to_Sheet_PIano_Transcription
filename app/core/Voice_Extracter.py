from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torchaudio
import torch.nn.functional as F
import yaml
from torch.amp import autocast

# ⚠️ model.py 위치에 맞게 import 경로 조정
from Seperate.src.MDX.model import TFC_TDF_net


def _dict_to_object(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_object(v) for k, v in d.items()})
    return d


def _load_config_from_yaml(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    return cfg_dict


def _load_audio_with_config(path: str, audio_cfg):
    """
    mp3/wav/flac 모두 torchaudio backend에 맡김.
    (Docker 배포 시 ffmpeg 설치 권장)
    """
    wav, sr = torchaudio.load(path)  # (C, T)

    target_sr = getattr(audio_cfg, "sample_rate", sr)
    num_channels = getattr(audio_cfg, "num_channels", wav.shape[0])
    chunk_size = getattr(audio_cfg, "chunk_size", None)

    # 채널 맞추기
    if wav.shape[0] < num_channels:
        rep = num_channels // wav.shape[0]
        wav = wav.repeat(rep, 1)
        wav = wav[:num_channels]
    elif wav.shape[0] > num_channels:
        wav = wav[:num_channels]

    # 리샘플
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    original_len = wav.shape[1]

    # chunk_size에 맞춰 pad
    if chunk_size is not None:
        pad = (-wav.shape[1]) % chunk_size
        if pad > 0:
            wav = F.pad(wav, (0, pad))

    wav = wav.unsqueeze(0)  # (1, C, T)
    return wav.float(), sr, original_len


def _run_inference_chunked(model, audio, device: torch.device, chunk_size: int):
    audio = audio.to(device)
    b, c, t = audio.shape
    if b != 1:
        raise ValueError(f"Expected batch=1, got {b}")

    if t % chunk_size != 0:
        raise ValueError(f"Audio length {t} must be divisible by chunk_size {chunk_size}")

    num_chunks = t // chunk_size
    out_chunks = []

    model.eval()
    with torch.no_grad():
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = audio[:, :, start:end]

            if device.type == "cuda":
                with autocast(device_type="cuda"):
                    out_chunk = model(chunk)
            else:
                out_chunk = model(chunk)

            out_chunks.append(out_chunk.detach().cpu())

    out = out_chunks[0]
    for i in range(1, len(out_chunks)):
        out = torch.cat([out, out_chunks[i]], dim=-1)

    return out.squeeze()


@dataclass
class VoiceExtractorConfig:
    ckpt_path: str
    yaml_path: str
    device: Optional[str] = None  # "cuda" | "cpu" | None(auto)


class VoiceExtractor:
    """
    MDX(TFC_TDF_net) 기반 2-stem 분리기.
    - 서버 시작 시 1회 로드해서 재사용
    - extract_instrumental_to_wav: Instrumental(=no vocals)만 저장
    - extract_instrumental_bytes: wav bytes 반환(원하면)
    """

    def __init__(self, cfg: VoiceExtractorConfig):
        self.ckpt_path = str(cfg.ckpt_path)
        self.yaml_path = str(cfg.yaml_path)

        if cfg.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        # config + model load
        cfg_dict = _load_config_from_yaml(self.yaml_path)
        self.config = _dict_to_object(cfg_dict)

        # weights_only=True 권장
        state = torch.load(self.ckpt_path, map_location=self.device, weights_only=True)

        self.model = TFC_TDF_net(self.config, str(self.device)).to(self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        # 자주 쓰는 값 캐싱
        self.audio_cfg = self.config.audio
        self.chunk_size = getattr(self.audio_cfg, "chunk_size", None)
        if self.chunk_size is None:
            raise ValueError("config.audio.chunk_size is required for chunked inference")

        self.stem_names = list(getattr(self.config.training, "instruments", ["Vocals", "Instrumental"]))

    def _select_stem(self, out: torch.Tensor, stem_name: str) -> torch.Tensor:
        """
        out expected:
          - (num_stems, C, T) or (C, T) or (T,)
        """
        if out.dim() == 3:
            stem_idx = self.stem_names.index(stem_name) if stem_name in self.stem_names else 0
            out = out[stem_idx]  # (C, T) or (T,)
        return out

    def extract_instrumental_to_wav(
        self,
        input_audio_path: str,
        output_wav_path: str,
        stem_name: str = "Instrumental",
    ) -> str:
        """
        input_audio_path: mp3/wav/flac path
        output_wav_path: output wav path
        stem_name: 보통 "Instrumental"
        """
        input_audio_path = str(input_audio_path)
        output_wav_path = str(output_wav_path)

        audio, sr, original_len = _load_audio_with_config(input_audio_path, self.audio_cfg)

        out = _run_inference_chunked(self.model, audio, self.device, self.chunk_size)
        out = self._select_stem(out, stem_name)

        # shape normalize to (C, T)
        if out.dim() == 1:
            out_to_save = out.unsqueeze(0)
        elif out.dim() == 2:
            out_to_save = out
        else:
            raise ValueError(f"Unexpected output dim: {out.dim()}")

        # pad 전 길이로 자르기
        out_to_save = out_to_save[..., :original_len]

        Path(output_wav_path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_wav_path, out_to_save, sr)

        return output_wav_path

    def extract_instrumental_bytes(
        self,
        input_audio_path: str,
        stem_name: str = "Instrumental",
    ) -> Tuple[bytes, int]:
        """
        instrumental wav bytes 반환 (sr은 필요하면 확장)
        """
        import tempfile

        tmp_dir = Path(tempfile.gettempdir()) / "s2s_mdx_inst"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_out = tmp_dir / f"inst_{Path(input_audio_path).stem}.wav"

        out_path = self.extract_instrumental_to_wav(
            input_audio_path=str(input_audio_path),
            output_wav_path=str(tmp_out),
            stem_name=stem_name,
        )
        wav_bytes = Path(out_path).read_bytes()
        return wav_bytes, 0
