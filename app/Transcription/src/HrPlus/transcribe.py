# test_hrplus_transcribe.py
# 1) wav 경로 하드코딩
# 2) make_cqt_dataset.py 와 100% 동일한 방식으로 CQT 생성
# 3) HRPlus 모델 로드 후 추론
# 4) onset/frame/offset 기반으로 노트 디코딩
# 5) MIDI 저장

import os
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import pretty_midi

from model import HRPlus   # 너의 HRPlus 모델

# ===========================
# 하드코딩 구간
# ===========================

AUDIO_PATH = r"C:\Users\hyi8402\Desktop\CQT\이.mp3"
CKPT_PATH  = r"C:\Users\hyi8402\Desktop\Sound to Sheet\app\Transcription\2.result_hrplus\hrplus_best.pt"

OUTPUT_MIDI = os.path.splitext(AUDIO_PATH)[0] + "_hrplus.mid"

SR   = 16000
HOP  = 512
FRAME_TIME = HOP / SR
N_BINS = 88
FMIN = librosa.note_to_hz("A0")   # 동일한 fmin 사용
MIDI_LOW = 21     # A0
MIDI_HIGH = 108   # C8

# 🔹 청크 길이 (초단위)
CHUNK_SEC = 20.0   # 20초씩 자르기

# 디코딩 threshold (기본값)
ONSET_TH      = 0.3
FRAME_ON_TH   = 0.25
FRAME_OFF_TH  = 0.15
MIN_DUR_SEC   = 0.03

# 🔹 피치가 높을수록 onset 기준을 얼마나 완화할지 (최대 감소량)
PITCH_RELAX_MAX = 0  # 맨 위 음역은 ONSET_TH - 0.05까지 허용


# ===========================
# 1) 오디오 → CQT (학습 CQT와 100% 동일)
# ===========================
def audio_to_cqt_tensor(path: Path):
    y, sr = librosa.load(path.as_posix(), sr=SR, mono=True)

    cqt = librosa.cqt(
        y,
        sr=SR,
        hop_length=HOP,
        fmin=FMIN,
        n_bins=N_BINS,
        bins_per_octave=12
    )
    cqt_mag = np.abs(cqt).astype(np.float32)        # (n_bins, T)
    cqt_tensor = torch.from_numpy(cqt_mag).unsqueeze(0)  # (1, F, T)
    return cqt_tensor.unsqueeze(0)  # (1, 1, F, T)


# ===========================
# 2) 모델 로드
# ===========================
def load_model(ckpt_path, device):
    model = HRPlus(
        n_pitches=88,
        cqt_bins=88,
        in_channels=1,
        base_channels=64,
        gru_hidden=128,
        pool_freq=1,   # 학습과 동일
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ===========================
# 3-A) 헬퍼: local max / parabolic refinement
# ===========================
def is_local_max(col, t):
    """1D array col에서 t가 양끝이 아니고 양 옆보다 크거나 같은지"""
    if t <= 0 or t >= len(col) - 1:
        return False
    return (col[t] >= col[t - 1]) and (col[t] >= col[t + 1])


def refine_peak_time(col, t):
    """
    onset 확률 시퀀스 col (shape: (T,))에서
    t-1, t, t+1 세 점으로 1D parabola 피팅해서
    서브프레임 정밀도 t_refined 반환
    """
    T = len(col)
    if t <= 0 or t >= T - 1:
        return float(t)

    y1, y2, y3 = float(col[t - 1]), float(col[t]), float(col[t + 1])
    denom = (y1 - 2.0 * y2 + y3)
    if abs(denom) < 1e-8:
        return float(t)

    delta = 0.5 * (y1 - y3) / denom  # 보통 -0.5~+0.5 근처
    # 너무 튀는 경우 방지
    if delta < -1.0:
        delta = -1.0
    elif delta > 1.0:
        delta = 1.0

    return float(t + delta)


# ===========================
# 3-B) 노트 추출 로직 (HR용 디코더)
# ===========================
def extract_notes(onset_p, frame_p, on_th, fr_on_th, fr_off_th, min_dur):
    """
    onset_p:  (T, 88) sigmoid된 확률 (numpy)
    frame_p:  (T, 88)
    """
    T, K = onset_p.shape

    active_notes = {}   # pitch(k) -> onset_time(float)
    notes = []

    for t in range(T):
        # onset 감지
        for k in range(K):
            col = onset_p[:, k]  # 특정 피치 k 의 전체 시퀀스 (T,)

            # 피치별 가변 threshold: 고음일수록 조금 더 관대
            midi_pitch = MIDI_LOW + k
            pitch_ratio = (midi_pitch - MIDI_LOW) / (MIDI_HIGH - MIDI_LOW)
            relax = PITCH_RELAX_MAX * pitch_ratio
            dyn_th = max(0.05, on_th - relax)  # 너무 낮아지지 않게 하한선

            # local max + threshold 조건 + 이미 켜진 음은 다시 시작 X
            if (
                onset_p[t, k] >= dyn_th
                and is_local_max(col, t)
                and k not in active_notes
            ):
                t_refined = refine_peak_time(col, t)  # 파라볼릭 보정
                onset_time = t_refined * FRAME_TIME
                active_notes[k] = onset_time

        # frame-based note 종료 판정
        for k in list(active_notes.keys()):
            p = frame_p[t, k]
            if p < fr_off_th:
                on_time = active_notes[k]
                off_time = t * FRAME_TIME

                if off_time - on_time >= min_dur:
                    notes.append((on_time, off_time, k))
                del active_notes[k]

    # 끝에 남은 노트 정리
    for k, on_time in active_notes.items():
        off_time = T * FRAME_TIME
        if off_time - on_time >= min_dur:
            notes.append((on_time, off_time, k))

    return notes


# ===========================
# 4) MIDI로 저장
# ===========================
def save_as_midi(notes, midi_path):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)

    for on, off, k in notes:
        pitch = MIDI_LOW + k
        n = pretty_midi.Note(
            velocity=90,
            pitch=pitch,
            start=on,
            end=off
        )
        inst.notes.append(n)

    pm.instruments.append(inst)
    pm.write(midi_path)
    print(f"[MIDI SAVED] {midi_path}")


# ===========================
# MAIN (청크 단위 추론)
# ===========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # 전체 CQT
    feat = audio_to_cqt_tensor(Path(AUDIO_PATH)).to(device)  # (1,1,88,T_total)
    print("CQT shape:", feat.shape)
    _, _, _, T_total = feat.shape

    # 청크 당 프레임 수
    chunk_frames = int(CHUNK_SEC / FRAME_TIME)
    print(f"Chunk frames: {chunk_frames} (≈ {CHUNK_SEC:.1f} sec)")

    # 모델
    model = load_model(CKPT_PATH, device)

    all_notes = []

    with torch.no_grad():
        start_frame = 0
        while start_frame < T_total:
            end_frame = min(start_frame + chunk_frames, T_total)
            feat_chunk = feat[:, :, :, start_frame:end_frame]  # (1,1,88,T_chunk)
            T_chunk = end_frame - start_frame

            if T_chunk <= 0:
                break

            print(f"Processing frames {start_frame} ~ {end_frame} (T_chunk={T_chunk})")

            out = model(feat_chunk)
            onset_p = torch.sigmoid(out["onset_logits"])[0].cpu().numpy()  # (T_chunk, 88)
            frame_p = torch.sigmoid(out["frame_logits"])[0].cpu().numpy()

            # 이 chunk 내부(0~T_chunk*FRAME_TIME 기준)에서 노트 추출
            notes_chunk = extract_notes(
                onset_p, frame_p,
                ONSET_TH, FRAME_ON_TH, FRAME_OFF_TH,
                MIN_DUR_SEC
            )

            # chunk 시작 시간(sec)만큼 더해서 전체 타임으로 이동
            chunk_start_sec = start_frame * FRAME_TIME
            for on, off, k in notes_chunk:
                all_notes.append((on + chunk_start_sec, off + chunk_start_sec, k))

            start_frame = end_frame

    print(f"Total detected notes: {len(all_notes)}")

    # MIDI 저장
    save_as_midi(all_notes, OUTPUT_MIDI)


if __name__ == "__main__":
    main()
