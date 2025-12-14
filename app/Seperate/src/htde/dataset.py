# dataset.py

from pathlib import Path
from typing import List, Tuple
from collections import deque

import torch
from torch.utils.data import Dataset


class SlakhPianoDataset(Dataset):
    """
    Slakh2100에서 mix/piano 파형을 불러와
    일정 길이(segment_seconds)로 잘라주는 Dataset.

    - .pt 파일 구조 가정: {"mix": Tensor(T,), "piano": Tensor(T,), "sr": int(옵션)}
    - segment_seconds: 한 세그먼트 길이(초)
    - hop_seconds: 세그먼트 간 간격(초)  (segment_seconds와 같으면 overlap 없음)
    - min_piano_rms: 피아노 RMS가 너무 작은 구간은 학습에서 제외 (노이즈/무음 필터링)
    - cache_tracks: 디스크 I/O 줄이기 위해 track 단위 캐시 사용 여부
    - preload_all: split에 있는 모든 .pt를 통째로 RAM에 올릴지 여부
      (메모리 여유 많을 때 True로 하면 torch.load 병목이 크게 줄어듦)
    - max_cache_tracks: preload_all=False일 때, LRU 캐시용으로 메모리에 유지할 트랙 수
    """

    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        segment_seconds: float = 6.0,
        hop_seconds: float = 3.0,
        target_sr: int = 22050,
        min_piano_rms: float = 0.0,
        cache_tracks: bool = True,
        preload_all: bool = False,
        max_cache_tracks: int = 32,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.split = split
        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.target_sr = target_sr
        self.min_piano_rms = min_piano_rms

        self.cache_tracks = cache_tracks
        self.preload_all = preload_all
        self.max_cache_tracks = max_cache_tracks

        # 세그먼트 길이/홉 (샘플 단위)
        self.seg_len = int(segment_seconds * target_sr)
        self.hop_len = int(hop_seconds * target_sr)

        # pt 파일 목록 수집 (하위 디렉토리까지 포함)
        split_dir = self.root_dir / self.split
        self.pt_paths: List[Path] = sorted(split_dir.rglob("*.pt"))
        if len(self.pt_paths) == 0:
            raise FileNotFoundError(f"No .pt files found in {split_dir}")

        print(f"[SlakhPianoDataset] split={split}, found {len(self.pt_paths)} pt files.")

        # preload / cache용 구조
        self.tracks: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._cache = {}              # {pt_idx: (mix, piano)}
        self._cache_order = deque()   # LRU 순서 (왼쪽이 가장 오래된 idx)

        # 인덱스: (pt_idx, start_sample)
        self.index: List[Tuple[int, int]] = []

        # ===== 인덱스 & (옵션) preload 생성 =====
        total_bytes = 0
        print(f"[SlakhPianoDataset] building index for split={split} ...")

        for i, pt_path in enumerate(self.pt_paths):
            # 한 트랙의 전체 파형 로드
            data = torch.load(pt_path, map_location="cpu")
            mix = data["mix"]         # (T,) 또는 (1,T)
            piano = data["piano"]     # (T,) 또는 (1,T)

            # mono (T,) 가정, 혹시 (1,T)면 squeeze
            if mix.dim() == 2:
                mix = mix.squeeze(0)
            if piano.dim() == 2:
                piano = piano.squeeze(0)

            T = mix.shape[-1]

            # preload_all이면 RAM에 올라갈 리스트에 저장
            if self.preload_all:
                self.tracks.append((mix, piano))
                total_bytes += (mix.numel() + piano.numel()) * 4  # float32 기준 4byte

            # 이 트랙에서 segment/hop으로 잘라 index 생성
            start = 0
            while start + self.seg_len <= T:
                end = start + self.seg_len
                if min_piano_rms > 0.0:
                    piano_seg = piano[start:end]
                    rms = (piano_seg ** 2).mean().item()
                    if rms < min_piano_rms:
                        # 너무 작은 RMS면 스킵
                        start += self.hop_len
                        continue

                self.index.append((i, start))
                start += self.hop_len

        print(f"[SlakhPianoDataset] total segments: {len(self.index)}")

        if self.preload_all:
            gb = total_bytes / (1024 ** 3)
            print(f"[SlakhPianoDataset] Preloaded {len(self.tracks)} tracks "
                  f"into RAM (~{gb:.2f} GB)")

    def __len__(self) -> int:
        return len(self.index)

    # ================================
    # 내부: 트랙 로딩 (preload / cache / 디스크)
    # ================================
    def _load_track(self, pt_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        pt_idx에 해당하는 (mix, piano) 파형을 반환.
        - preload_all=True: self.tracks에서 바로 꺼냄 (디스크 I/O 없음)
        - preload_all=False & cache_tracks=True: LRU 캐시 활용
        - 그 외: torch.load로 .pt에서 직접 로드
        """
        # 1) 전체 preload 모드면 그냥 RAM에서 꺼냄
        if self.preload_all:
            mix, piano = self.tracks[pt_idx]
            return mix, piano

        # 2) LRU 캐시 hit
        if self.cache_tracks and pt_idx in self._cache:
            mix, piano = self._cache[pt_idx]
            # 최근 사용으로 갱신
            try:
                self._cache_order.remove(pt_idx)
            except ValueError:
                pass
            self._cache_order.append(pt_idx)
            return mix, piano

        # 3) 디스크에서 로드
        pt_path = self.pt_paths[pt_idx]
        data = torch.load(pt_path, map_location="cpu")
        mix = data["mix"]
        piano = data["piano"]

        if mix.dim() == 2:
            mix = mix.squeeze(0)
        if piano.dim() == 2:
            piano = piano.squeeze(0)

        # 4) LRU 캐시에 넣기
        if self.cache_tracks:
            if len(self._cache_order) >= self.max_cache_tracks:
                # 가장 오래된 트랙 하나 제거
                old_idx = self._cache_order.popleft()
                if old_idx in self._cache:
                    del self._cache[old_idx]

            self._cache[pt_idx] = (mix, piano)
            self._cache_order.append(pt_idx)

        return mix, piano

    # ================================
    # __getitem__
    # ================================
    def __getitem__(self, idx: int):
        pt_idx, start = self.index[idx]
        mix, piano = self._load_track(pt_idx)

        end = start + self.seg_len
        mix_seg = mix[start:end]       # (T,)
        piano_seg = piano[start:end]   # (T,)

        # DataLoader에서 batch_first로 묶이면 (B, T) 형태가 됨
        return mix_seg, piano_seg
