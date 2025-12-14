# data_utils.py (HRplus 스타일 라벨 버전)

import torch

# ==========================
# 기본 설정
# ==========================
SR = 16000
HOP = 512              # CQT 만들 때 사용한 hop_length와 반드시 동일해야 함
FRAME_TIME = HOP / SR  # 0.032 sec (HOP=512 기준)

# 피아노 범위 (필요에 따라 바꿔도 됨)
MIDI_LOW = 21    # A0
MIDI_HIGH = 108  # C8


# ==========================
# 1. 노트 → HRplus 라벨 행렬 변환
# ==========================
def notes_to_hrplus_labels(
    notes,
    n_frames,
    frame_time: float = FRAME_TIME,
    midi_low: int = MIDI_LOW,
    midi_high: int = MIDI_HIGH,
    J: int = 3,   # 논문에서 J=5 (±5프레임 영향)
):
    """
    HRplus / High-resolution 스타일 라벨 생성:
      - onset_hr:  (T, K)  연속값 g(Δ)
      - offset_hr: (T, K)  연속값 g(Δ)
      - frame_bin:(T, K)  binary (note active 여부)

    notes: list of (onset_sec, offset_sec, pitch)
    n_frames: 세그먼트 내 프레임 수 (CQT time axis 길이)
    """
    device = torch.device("cpu")

    n_keys = midi_high - midi_low + 1

    onset_hr  = torch.zeros(n_frames, n_keys, dtype=torch.float32, device=device)
    offset_hr = torch.zeros(n_frames, n_keys, dtype=torch.float32, device=device)
    frame_bin = torch.zeros(n_frames, n_keys, dtype=torch.float32, device=device)

    # 각 프레임의 시간 (단순히 t * frame_time 사용)
    frame_times = torch.arange(n_frames, dtype=torch.float32, device=device) * frame_time
    J_delta = J * frame_time

    eps = 1e-8

    for onset, offset, pitch in notes:
        if pitch < midi_low or pitch > midi_high:
            continue
        key_idx = pitch - midi_low

        # ========== frame (binary) ==========
        on_f = int(round(onset / frame_time))
        off_f = int(round(offset / frame_time))

        on_f = max(0, min(on_f,  n_frames - 1))
        off_f = max(0, min(off_f, n_frames))

        if off_f <= on_f:
            continue

        frame_bin[on_f:off_f, key_idx] = 1.0

        # ========== onset high-resolution g(Δ) ==========
        center_on = onset
        delta_on = torch.abs(frame_times - center_on)
        mask_on = delta_on <= J_delta
        if mask_on.any():
            g_on = 1.0 - (delta_on[mask_on] / (J_delta + eps))
            # 기존 값과 max를 취해서 여러 노트가 겹칠 때도 안정적으로 유지
            onset_hr[mask_on, key_idx] = torch.max(onset_hr[mask_on, key_idx], g_on)

        # ========== offset high-resolution g(Δ) ==========
        center_off = offset
        delta_off = torch.abs(frame_times - center_off)
        mask_off = delta_off <= J_delta
        if mask_off.any():
            g_off = 1.0 - (delta_off[mask_off] / (J_delta + eps))
            offset_hr[mask_off, key_idx] = torch.max(offset_hr[mask_off, key_idx], g_off)

    return onset_hr, offset_hr, frame_bin


# ==========================
# 2. 개별 샘플 준비
# ==========================
def prepare_sample_for_training(
    feat_chunk: torch.Tensor,
    notes,
    frame_time: float = FRAME_TIME,
    midi_low: int = MIDI_LOW,
    midi_high: int = MIDI_HIGH,
):
    """
    세그먼트 단위 CQT + 노트 리스트를
    HRplus 라벨 텐서로 변환.

    feat_chunk : (1, n_bins, T_seg)  ← CQT 세그먼트
    notes      : list of (onset_local_sec, offset_local_sec, pitch)

    return:
        feat_chunk:  (1, n_bins, T_seg)  ← 그대로 반환
        onset_hr:   (T_seg, n_keys)
        offset_hr:  (T_seg, n_keys)
        frame_bin:  (T_seg, n_keys)
    """
    n_frames = feat_chunk.shape[-1]

    onset_hr, offset_hr, frame_bin = notes_to_hrplus_labels(
        notes,
        n_frames=n_frames,
        frame_time=frame_time,
        midi_low=midi_low,
        midi_high=midi_high,
    )

    return feat_chunk, onset_hr, offset_hr, frame_bin


# ==========================
# 3. collate_fn (batch 구성)
# ==========================
def collate_hrplus(batch):
    """
    batch: list of (feat_chunk, notes, stem, txt_name)
           - feat_chunk: (1, n_bins, T_i)
           - notes     : list of (onset_local_sec, offset_local_sec, pitch)
           - stem      : 곡 이름 (ex: AkPnBcht_01)
           - txt_name  : 라벨 파일 이름 (ex: AkPnBcht_01_000.txt)

    return:
        feat_batch:   (B, 1, n_bins, T_max)
        onset_batch:  (B, T_max, n_keys)
        offset_batch: (B, T_max, n_keys)
        frame_batch:  (B, T_max, n_keys)
        lengths:      (B,)  각 샘플의 실제 길이(T_i)
        metas:        list of (stem, txt_name)
    """
    feat_list = []
    onset_list = []
    offset_list = []
    frame_list = []
    lengths = []
    metas = []

    for feat_chunk, notes, stem, txt_name in batch:
        feat_chunk, onset_hr, offset_hr, frame_bin = prepare_sample_for_training(
            feat_chunk, notes
        )
        T = feat_chunk.shape[-1]

        feat_list.append(feat_chunk)
        onset_list.append(onset_hr)
        offset_list.append(offset_hr)
        frame_list.append(frame_bin)
        lengths.append(T)
        metas.append((stem, txt_name))

    B = len(batch)
    n_bins = feat_list[0].shape[1]
    n_keys = onset_list[0].shape[1]
    T_max = max(lengths)

    # 패딩용 배치 텐서 초기화
    feat_batch   = torch.zeros(B, 1, n_bins, T_max, dtype=torch.float32)
    onset_batch  = torch.zeros(B, T_max, n_keys,  dtype=torch.float32)
    offset_batch = torch.zeros(B, T_max, n_keys,  dtype=torch.float32)
    frame_batch  = torch.zeros(B, T_max, n_keys,  dtype=torch.float32)
    lengths_t    = torch.tensor(lengths, dtype=torch.long)

    for i in range(B):
        T = lengths[i]
        feat_batch[i, :, :, :T] = feat_list[i]
        onset_batch[i, :T, :]   = onset_list[i]
        offset_batch[i, :T, :]  = offset_list[i]
        frame_batch[i, :T, :]   = frame_list[i]

    return feat_batch, onset_batch, offset_batch, frame_batch, lengths_t, metas
