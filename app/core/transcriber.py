# core/transcriber.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import io
import tempfile

import librosa
import numpy as np
import torch
import pretty_midi

from Transcription.src.HrPlus.model import HRPlus


@dataclass
class DecodeConfig:
    SR: int = 16000
    HOP: int = 512
    N_BINS: int = 88
    FMIN: float = librosa.note_to_hz("A0")
    MIDI_LOW: int = 21
    MIDI_HIGH: int = 108

    CHUNK_SEC: float = 20.0

    ONSET_TH: float = 0.3
    FRAME_ON_TH: float = 0.25  # (현재 로직에선 직접 쓰진 않지만 유지)
    FRAME_OFF_TH: float = 0.15
    MIN_DUR_SEC: float = 0.03

    PITCH_RELAX_MAX: float = 0.0  # 필요하면 0.05 같은 값으로


class HRPlusTranscriber:
    def __init__(self, ckpt_path: Path, device: torch.device, cfg: Optional[DecodeConfig] = None):
        self.ckpt_path = Path(ckpt_path)
        self.device = device
        self.cfg = cfg or DecodeConfig()

        self.frame_time = self.cfg.HOP / self.cfg.SR

        self.model = HRPlus(
            n_pitches=88,
            cqt_bins=88,
            in_channels=1,
            base_channels=64,
            gru_hidden=128,
            pool_freq=1,
        ).to(self.device)

        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()

    # ---------- CQT ----------
    def audio_to_cqt_tensor(self, path: Path) -> torch.Tensor:
        y, _ = librosa.load(path.as_posix(), sr=self.cfg.SR, mono=True)

        cqt = librosa.cqt(
            y,
            sr=self.cfg.SR,
            hop_length=self.cfg.HOP,
            fmin=self.cfg.FMIN,
            n_bins=self.cfg.N_BINS,
            bins_per_octave=12,
        )
        cqt_mag = np.abs(cqt).astype(np.float32)              # (F, T)
        cqt_tensor = torch.from_numpy(cqt_mag).unsqueeze(0)   # (1, F, T)
        return cqt_tensor.unsqueeze(0)                        # (1, 1, F, T)

    # ---------- Peak helpers ----------
    @staticmethod
    def is_local_max(col: np.ndarray, t: int) -> bool:
        if t <= 0 or t >= len(col) - 1:
            return False
        return (col[t] >= col[t - 1]) and (col[t] >= col[t + 1])

    @staticmethod
    def refine_peak_time(col: np.ndarray, t: int) -> float:
        T = len(col)
        if t <= 0 or t >= T - 1:
            return float(t)
        y1, y2, y3 = float(col[t - 1]), float(col[t]), float(col[t + 1])
        denom = (y1 - 2.0 * y2 + y3)
        if abs(denom) < 1e-8:
            return float(t)
        delta = 0.5 * (y1 - y3) / denom
        delta = max(-1.0, min(1.0, delta))
        return float(t + delta)

    # ---------- Decode ----------
    def extract_notes(
        self,
        onset_p: np.ndarray,   # (T, 88)
        frame_p: np.ndarray,   # (T, 88)
    ) -> List[Tuple[float, float, int]]:
        cfg = self.cfg
        T, K = onset_p.shape

        active = {}  # k -> onset_time_sec(float)
        notes = []

        for t in range(T):
            for k in range(K):
                col = onset_p[:, k]

                midi_pitch = cfg.MIDI_LOW + k
                pitch_ratio = (midi_pitch - cfg.MIDI_LOW) / (cfg.MIDI_HIGH - cfg.MIDI_LOW)
                relax = cfg.PITCH_RELAX_MAX * pitch_ratio
                dyn_th = max(0.05, cfg.ONSET_TH - relax)

                if (
                    onset_p[t, k] >= dyn_th
                    and self.is_local_max(col, t)
                    and k not in active
                ):
                    t_ref = self.refine_peak_time(col, t)
                    active[k] = t_ref * self.frame_time

            for k in list(active.keys()):
                if frame_p[t, k] < cfg.FRAME_OFF_TH:
                    on_time = active[k]
                    off_time = t * self.frame_time
                    if off_time - on_time >= cfg.MIN_DUR_SEC:
                        notes.append((on_time, off_time, k))
                    del active[k]

        for k, on_time in active.items():
            off_time = T * self.frame_time
            if off_time - on_time >= cfg.MIN_DUR_SEC:
                notes.append((on_time, off_time, k))

        return notes

    def _notes_to_pretty_midi(self, notes: List[Tuple[float, float, int]]) -> pretty_midi.PrettyMIDI:
        cfg = self.cfg
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)

        for on, off, k in notes:
            pitch = cfg.MIDI_LOW + k
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=pitch, start=on, end=off))

        pm.instruments.append(inst)
        return pm

    def save_as_midi(self, notes: List[Tuple[float, float, int]], midi_path: Path) -> None:
        pm = self._notes_to_pretty_midi(notes)
        pm.write(midi_path.as_posix())

    def save_as_midi_bytes(self, notes: List[Tuple[float, float, int]]) -> bytes:
        """
        pretty_midi 버전에 따라 BytesIO 직접 write가 안 될 수 있어서:
        1) BytesIO 시도
        2) 실패하면 NamedTemporaryFile에 썼다가 읽어서 bytes 반환
        """
        pm = self._notes_to_pretty_midi(notes)

        # 1) try BytesIO
        try:
            buf = io.BytesIO()
            pm.write(buf)  # 일부 환경에서 지원됨
            return buf.getvalue()
        except Exception:
            pass

        # 2) fallback: temp file
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=True) as tf:
            pm.write(tf.name)
            tf.seek(0)
            return tf.read()

    # ---------- Public API ----------
    @torch.inference_mode()
    def transcribe_to_midi(self, audio_path: Path, out_midi_path: Path) -> Path:
        feat = self.audio_to_cqt_tensor(audio_path).to(self.device)  # (1,1,88,T)
        _, _, _, T_total = feat.shape

        chunk_frames = int(self.cfg.CHUNK_SEC / self.frame_time)
        all_notes: List[Tuple[float, float, int]] = []

        start = 0
        while start < T_total:
            end = min(start + chunk_frames, T_total)
            feat_chunk = feat[:, :, :, start:end]
            if end - start <= 0:
                break

            out = self.model(feat_chunk)
            onset_p = torch.sigmoid(out["onset_logits"])[0].detach().cpu().numpy()  # (T,88)
            frame_p = torch.sigmoid(out["frame_logits"])[0].detach().cpu().numpy()

            notes_chunk = self.extract_notes(onset_p, frame_p)

            offset_sec = start * self.frame_time
            for on, off, k in notes_chunk:
                all_notes.append((on + offset_sec, off + offset_sec, k))

            start = end

        self.save_as_midi(all_notes, out_midi_path)
        return out_midi_path

    @torch.inference_mode()
    def transcribe_to_midi_bytes(self, audio_path: Path) -> bytes:
        feat = self.audio_to_cqt_tensor(audio_path).to(self.device)  # (1,1,88,T)
        _, _, _, T_total = feat.shape

        chunk_frames = int(self.cfg.CHUNK_SEC / self.frame_time)
        all_notes: List[Tuple[float, float, int]] = []

        start = 0
        while start < T_total:
            end = min(start + chunk_frames, T_total)
            feat_chunk = feat[:, :, :, start:end]
            if end - start <= 0:
                break

            out = self.model(feat_chunk)
            onset_p = torch.sigmoid(out["onset_logits"])[0].detach().cpu().numpy()  # (T,88)
            frame_p = torch.sigmoid(out["frame_logits"])[0].detach().cpu().numpy()

            notes_chunk = self.extract_notes(onset_p, frame_p)

            offset_sec = start * self.frame_time
            for on, off, k in notes_chunk:
                all_notes.append((on + offset_sec, off + offset_sec, k))

            start = end

        return self.save_as_midi_bytes(all_notes)
