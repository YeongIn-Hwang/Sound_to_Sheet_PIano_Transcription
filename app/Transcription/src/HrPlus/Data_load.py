# Data_load.py (CQT 세그먼트 로더)

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

SR = 16000
HOP = 512
FRAME_TIME = HOP / SR   # 0.032 sec  ← CQT hop_length와 동일해야 함


# ==========================================================
# Utility: Parse segment header
# ==========================================================
def parse_segment_header(txt_path):
    """
    TXT 첫줄:
    # SEGMENT_INFO: START=40.123456 END=60.654321
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()

    # "# SEGMENT_INFO: START=40.123 END=60.321"
    parts = header.split()
    t_start = float(parts[2].split("=")[1])
    t_end   = float(parts[3].split("=")[1])
    return t_start, t_end


# ==========================================================
# Utility: Load labels (local time) from txt
# ==========================================================
def load_labels(txt_path):
    """
    세그먼트 기준 local time 라벨 로드
    (헤더 한 줄 + 컬럼 헤더 한 줄 이후의 (Onset, Offset, Pitch))
    """
    notes = []
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]  # 첫 줄 헤더(# SEGMENT_INFO)는 건너뜀
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Onset"):
            continue
        on, off, pitch = line.split()
        notes.append((float(on), float(off), int(pitch)))
    return notes


# ==========================================================
# LRU Feature CACHE (CQT / mel 공용)
# ==========================================================
class FeatureCache:
    def __init__(self, feat_dir, max_ram_gb=10):
        """
        feat_dir: stem.pt 들이 들어있는 디렉터리 (CQT든 mel이든 상관없음)
        """
        self.feat_dir = Path(feat_dir)
        self.cache = OrderedDict()  # stem -> tensor
        self.max_ram_bytes = max_ram_gb * (1024**3)
        self.used_bytes = 0

    def _estimate_tensor_bytes(self, tensor):
        return tensor.numel() * tensor.element_size()

    def _evict_if_needed(self):
        # 메모리 제한 넘으면 오래된 것부터 제거
        while self.used_bytes > self.max_ram_bytes and len(self.cache) > 0:
            old_stem, old_tensor = self.cache.popitem(last=False)  # FIFO
            old_size = self._estimate_tensor_bytes(old_tensor)
            self.used_bytes -= old_size
            # print(f"[CACHE] Evicted {old_stem} ({old_size/1e6:.1f} MB)")

    def load_feat(self, stem):
        # 이미 로드됨
        if stem in self.cache:
            feat = self.cache.pop(stem)
            self.cache[stem] = feat   # update to newest
            return feat

        # 새로 로드
        feat_path = self.feat_dir / f"{stem}.pt"
        if not feat_path.exists():
            raise FileNotFoundError(f"FEATURE NOT FOUND: {feat_path}")

        feat = torch.load(feat_path, weights_only=False)
        # (n_bins, T) 형태면 (1, n_bins, T)로 변환
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)

        size_bytes = self._estimate_tensor_bytes(feat)
        # print(f"[CACHE] Loading {stem} ({size_bytes/1e6:.1f} MB)")

        # 캐시에 저장
        self.cache[stem] = feat
        self.used_bytes += size_bytes

        # 메모리 초과 시 제거
        self._evict_if_needed()

        return feat


# ==========================================================
# DATASET
# ==========================================================
class HRSegmentDataset(Dataset):
    def __init__(self, label_dir, feat_dir, max_ram_gb=10):
        """
        label_dir: segment txt 모음
        feat_dir:  곡 단위 CQT(or mel) .pt 모음
        """

        self.label_paths = sorted(Path(label_dir).glob("*.txt"))
        self.feat_cache = FeatureCache(feat_dir, max_ram_gb)

        self.samples = []
        for txt_path in self.label_paths:
            # segment txt 이름 → stem 추출 (곡명)
            # ex) AkPnBcht_01_005 → stem = AkPnBcht_01
            base = txt_path.stem
            parts = base.split("_")
            stem = "_".join(parts[:-1])

            t_start, t_end = parse_segment_header(txt_path)

            self.samples.append({
                "txt": txt_path,
                "stem": stem,
                "t_start": t_start,
                "t_end": t_end,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        txt_path = sample["txt"]
        stem = sample["stem"]
        t_start = sample["t_start"]
        t_end   = sample["t_end"]

        # CQT (or mel) LOAD (LRU 캐시)
        feat = self.feat_cache.load_feat(stem)   # (1, n_bins, T_total)

        # 시간 → 프레임 인덱스 변환
        start_frame = int(round(t_start / FRAME_TIME))
        end_frame   = int(round(t_end   / FRAME_TIME))

        start_frame = max(0, min(start_frame, feat.shape[-1]))
        end_frame   = max(0, min(end_frame, feat.shape[-1]))

        feat_chunk = feat[:, :, start_frame:end_frame]  # (1, n_bins, T_seg)

        # labels (local time)
        notes = load_labels(txt_path)

        return feat_chunk, notes, stem, txt_path.name


# ==========================================================
# Example DataLoader
# ==========================================================
def get_loader(label_dir, feat_dir, batch=4, workers=2, max_ram_gb=8):
    dataset = HRSegmentDataset(label_dir, feat_dir, max_ram_gb=max_ram_gb)
    loader  = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    return loader
