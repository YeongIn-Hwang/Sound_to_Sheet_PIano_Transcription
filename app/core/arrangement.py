from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import io
import tempfile

import numpy as np
import torch

from arrangement.model import AccompHybridUNet


@dataclass
class AccompServiceConfig:
    ckpt_path: str

    device: Optional[str] = None  # "cuda" | "cpu" | None(auto)

    fs: int = 100
    segment_len: int = 512
    overlap: int = 256

    thresh: float = 0.6
    min_note_sec: float = 0.15
    out_velocity: int = 80


class AccompService:
    """
    FastAPI에서 서버 시작 시 1회 로드 후 재사용하는 반주 생성기.
    - generate_midi_bytes: 입력 MIDI bytes -> 출력 MIDI bytes
    - (필요하면) generate_to_file: 입력 midi path -> 출력 path
    """

    def __init__(self, cfg: AccompServiceConfig):
        self.cfg = cfg

        if cfg.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)

        # 모델 로드(1회)
        self.model = AccompHybridUNet().to(self.device)

        # ✅ 경고 방지: weights_only=True (state_dict만 로드하는 용도)
        state = torch.load(cfg.ckpt_path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"], strict=True)
        else:
            self.model.load_state_dict(state, strict=True)

        self.model.eval()

        self.frame_sec = 1.0 / float(cfg.fs)

    # ----------------------
    # MIDI <-> Roll
    # ----------------------
    def _midi_bytes_to_melody_roll(self, midi_bytes: bytes) -> np.ndarray:
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
        pr = pm.get_piano_roll(fs=self.cfg.fs)       # (128, T)
        pr = (pr > 0).astype(np.float32).T          # (T, 128)

        melody = np.zeros_like(pr)
        for t in range(pr.shape[0]):
            pitches = np.where(pr[t] > 0)[0]
            if len(pitches) > 0:
                melody[t, pitches.max()] = 1.0
        return melody

    @torch.no_grad()
    def _infer_probs(self, melody_roll: np.ndarray) -> np.ndarray:
        T = melody_roll.shape[0]
        probs_sum = np.zeros((T, 128), dtype=np.float32)
        weight = np.zeros((T, 128), dtype=np.float32)

        seg_len = self.cfg.segment_len
        step = max(1, seg_len - self.cfg.overlap)

        for start in range(0, T, step):
            seg = melody_roll[start:start + seg_len]
            if seg.shape[0] < seg_len:
                seg = np.pad(seg, ((0, seg_len - seg.shape[0]), (0, 0)))

            x = torch.from_numpy(seg[None, None]).to(self.device)  # (1,1,T,128)
            p = torch.sigmoid(self.model(x))[0, 0].detach().cpu().numpy()  # (T,128)

            valid = min(seg_len, T - start)
            probs_sum[start:start + valid] += p[:valid]
            weight[start:start + valid] += 1.0

        return probs_sum / np.maximum(weight, 1e-6)

    def _binary_roll_to_instrument(self, binary_roll: np.ndarray, velocity: int):
        import pretty_midi

        inst = pretty_midi.Instrument(program=0)
        T, P = binary_roll.shape
        min_frames = max(1, int(self.cfg.min_note_sec / self.frame_sec))

        for pitch in range(P):
            on = None
            run = 0
            for t in range(T):
                if binary_roll[t, pitch]:
                    if on is None:
                        on = t
                        run = 1
                    else:
                        run += 1
                elif on is not None:
                    if run >= min_frames:
                        inst.notes.append(
                            pretty_midi.Note(
                                velocity=velocity,
                                pitch=pitch,
                                start=on * self.frame_sec,
                                end=t * self.frame_sec,
                            )
                        )
                    on = None
                    run = 0

            if on is not None and run >= min_frames:
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=on * self.frame_sec,
                        end=T * self.frame_sec,
                    )
                )

        return inst

    # ----------------------
    # Public API
    # ----------------------
    def generate_midi_bytes(self, input_midi_bytes: bytes, keep_original: bool = True) -> bytes:
        """
        입력: MIDI bytes
        출력: (원본 유지 + 반주 트랙 추가) MIDI bytes
        """
        import pretty_midi
        import os
        import tempfile

        melody_roll = self._midi_bytes_to_melody_roll(input_midi_bytes)
        probs = self._infer_probs(melody_roll)
        binary = (probs >= self.cfg.thresh).astype(np.uint8)

        if keep_original:
            pm_out = pretty_midi.PrettyMIDI(io.BytesIO(input_midi_bytes))
        else:
            pm_out = pretty_midi.PrettyMIDI()

        pm_out.instruments.append(
            self._binary_roll_to_instrument(binary, velocity=self.cfg.out_velocity)
        )

        # ✅ Windows-safe temp file (핵심: fd 닫고, 그 경로에 pretty_midi가 write)
        fd, tmp_path = tempfile.mkstemp(suffix=".mid")
        os.close(fd)  # <- 이게 중요 (열린 핸들 때문에 Permission denied 나는 걸 방지)

        try:
            pm_out.write(tmp_path)
            return Path(tmp_path).read_bytes()
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

