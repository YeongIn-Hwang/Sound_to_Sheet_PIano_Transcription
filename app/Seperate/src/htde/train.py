# train.py
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
from tqdm.auto import tqdm  # 진행바

from dataset import SlakhPianoDataset
from model import HTDemucs

import gc


# ===========================
# 하드코딩 설정 (네가 바꿔서 쓰면 됨)
# ===========================
# 데이터 루트: 그 안에 train_piano / val_piano / test_piano 폴더가 있음
DATA_ROOT = Path(r"C:\Users\hyi8402\Desktop\Sound to Sheet\Dataset\slakh2100")

# 체크포인트 저장 폴더
CKPT_DIR = Path(r"C:\Users\hyi8402\Desktop\Sound to Sheet\app\Seperate\src\htde")

# 학습 하이퍼파라미터
NUM_EPOCHS      = 30
BATCH_SIZE      = 12
LR              = 3e-4
WEIGHT_DECAY    = 0.0
NUM_WORKERS     = 2

SEGMENT_SECONDS = 6.0    # 한 세그먼트 길이 (초)
HOP_SECONDS     = 3.0    # 세그먼트 간 간격 (초)
TARGET_SR       = 22050  # dataset.py에서 쓰는 target_sr
MIN_PIANO_RMS   = 0.01    # 피아노 에너지 필터링 (원하면 0.01~0.02 등으로 올려도 됨)

# Dataset 내부 LRU 캐시 관련 (RAM 절약용)
CACHE_TRACKS     = True
PRELOAD_ALL      = False        # 80GB 전체 preload 금지
MAX_CACHE_TRACKS = 128           # RAM 32GB 기준으로 적당히 (필요하면 늘리거나 줄여라)

# AMP / cudnn
USE_AMP         = True
cudnn.benchmark = True
cudnn.enabled   = True

# 이어학습 / 검증 사용 여부 (하드코딩)
RESUME_TRAIN   = True   # True면 last.pth 있으면 이어함
USE_VALIDATION = True  # False면 val 없이 train만

# Loss 가중치 (HTDemucs 스타일: time L1 + STFT L1 조합)
TIME_LOSS_WEIGHT = 1.0
STFT_LOSS_WEIGHT = 0.5  # 필요하면 1.0으로 올려도 됨

#####################################################################
# 전역 window 캐시 추가 
####################################################################

_STFT_WINDOWS = {}

def get_hann_window(win_len: int, device: torch.device):
    key = (win_len, str(device))
    w = _STFT_WINDOWS.get(key, None)
    if w is None or w.device != device:
        w = torch.hann_window(win_len, device=device)
        _STFT_WINDOWS[key] = w
    return w

# ===========================
# Loss 함수: waveform L1 + multi-resolution STFT L1
# ===========================
def _to_waveform(x: torch.Tensor) -> torch.Tensor:
    """
    pred/target 텐서를 (B, T) 형태로 변환.
    현재는 (B, 1, 1, T)로 들어오므로 다 squeeze.
    """
    if x.dim() == 4:
        # (B, S=1, C=1, T) -> (B, T)
        x = x.squeeze(2).squeeze(1)  # 두 번 squeeze
    elif x.dim() == 3:
        # (B, 1, T) -> (B, T)
        x = x.squeeze(1)
    # 그 외는 (B, T)라고 가정
    return x


def multi_resolution_stft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fft_sizes=(512, 1024, 2048),
    hop_sizes=(128, 256, 512),
    win_lengths=(512, 1024, 2048),
) -> torch.Tensor:
    """
    멀티 스케일 STFT 스펙트럼 L1 loss.
    pred, target: (B, 1, 1, T) 또는 (B, T)
    """
    pred_wav = _to_waveform(pred)
    target_wav = _to_waveform(target)

    if pred_wav.shape != target_wav.shape:
        min_len = min(pred_wav.shape[-1], target_wav.shape[-1])
        pred_wav = pred_wav[..., :min_len]
        target_wav = target_wav[..., :min_len]

    device = pred_wav.device
    total_loss = 0.0
    n_scales = len(fft_sizes)

    for n_fft, hop, win_len in zip(fft_sizes, hop_sizes, win_lengths):
        window = get_hann_window(win_len, device)

        # center=True로 STFT
        pred_stft = torch.stft(
            pred_wav,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win_len,
            window=window,
            center=True,
            return_complex=True,
        )
        target_stft = torch.stft(
            target_wav,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win_len,
            window=window,
            center=True,
            return_complex=True,
        )

        # magnitude L1 차이
        spec_loss = (pred_stft.abs() - target_stft.abs()).abs().mean()
        total_loss = total_loss + spec_loss

    return total_loss / n_scales


def htdemucs_style_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    HTDemucs 스타일:
    time-domain L1 + multi-resolution STFT L1 조합.
    """
    # time-domain L1 (wave L1)
    time_loss = (pred - target).abs().mean()

    # spectral L1
    spec_loss = multi_resolution_stft_loss(pred, target)

    loss = TIME_LOSS_WEIGHT * time_loss + STFT_LOSS_WEIGHT * spec_loss
    return loss, time_loss.detach(), spec_loss.detach()


# ===========================
# 유틸 함수
# ===========================
def create_dataloaders():
    """
    SlakhPianoDataset → DataLoader 생성
    train: train_piano
    val  : val_piano (USE_VALIDATION=True일 때만)
    """
    train_ds = SlakhPianoDataset(
        root_dir=DATA_ROOT,
        split="train_piano",
        segment_seconds=SEGMENT_SECONDS,
        hop_seconds=HOP_SECONDS,
        target_sr=TARGET_SR,
        min_piano_rms=MIN_PIANO_RMS,
        cache_tracks=CACHE_TRACKS,
        preload_all=PRELOAD_ALL,
        max_cache_tracks=MAX_CACHE_TRACKS,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    if USE_VALIDATION:
        val_ds = SlakhPianoDataset(
            root_dir=DATA_ROOT,
            split="val_piano",
            segment_seconds=SEGMENT_SECONDS,
            hop_seconds=HOP_SECONDS,  # val도 같은 세그멘트 정책
            target_sr=TARGET_SR,
            min_piano_rms=MIN_PIANO_RMS,
            cache_tracks=CACHE_TRACKS,
            preload_all=PRELOAD_ALL,
            max_cache_tracks=MAX_CACHE_TRACKS,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )
    else:
        val_loader = None

    return train_loader, val_loader


def save_checkpoint(epoch, model, optimizer, scaler, best_val_loss, ckpt_dir: Path):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
    }
    last_path = ckpt_dir / "last.pth"
    torch.save(state, last_path)

    # best는 밖에서 조건 체크 후 따로 저장
    return last_path


def load_checkpoint(model, optimizer, scaler, ckpt_dir: Path, device):
    last_path = ckpt_dir / "last.pth"
    if not last_path.exists():
        print(f"[Resume] No checkpoint found at {last_path}, starting from scratch.")
        return 1, float("inf")

    print(f"[Resume] Loading checkpoint from {last_path}")
    ckpt = torch.load(last_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"[Resume] Resuming from epoch {start_epoch}")
    return start_epoch, best_val_loss


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    use_amp: bool = True,
):
    model.train()
    running_loss = 0.0
    running_time_loss = 0.0
    running_spec_loss = 0.0
    num_batches = 0

    start_time = time.time()

    # tqdm 진행바
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", ncols=100)

    for batch_idx, (mix, piano) in enumerate(pbar):
        # mix, piano: (B, T)
        mix = mix.to(device, non_blocking=True)
        piano = piano.to(device, non_blocking=True)

        # mono 입력 기준 → (B, 1, T)
        mix = mix.unsqueeze(1)                       # (B, 1, T)
        target = piano.unsqueeze(1).unsqueeze(1)     # (B, 1, 1, T)  (sources=1, channels=1)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with autocast():
                pred = model(mix)                   # (B, S=1, C=1, T)
                loss, time_loss, spec_loss = htdemucs_style_loss(pred, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(mix)
            loss, time_loss, spec_loss = htdemucs_style_loss(pred, target)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_time_loss += float(time_loss)
        running_spec_loss += float(spec_loss)
        num_batches += 1

        avg_loss = running_loss / num_batches
        avg_time = running_time_loss / num_batches
        avg_spec = running_spec_loss / num_batches
        elapsed = time.time() - start_time

        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            time=f"{avg_time:.4f}",
            spec=f"{avg_spec:.4f}",
            sec=f"{elapsed:.0f}",
        )

    epoch_loss = running_loss / max(num_batches, 1)
    return epoch_loss


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    device,
    epoch,
):
    model.eval()
    running_loss = 0.0
    running_time_loss = 0.0
    running_spec_loss = 0.0
    num_batches = 0

    start_time = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", ncols=100)

    for batch_idx, (mix, piano) in enumerate(pbar):
        mix = mix.to(device, non_blocking=True)
        piano = piano.to(device, non_blocking=True)

        mix = mix.unsqueeze(1)                       # (B, 1, T)
        target = piano.unsqueeze(1).unsqueeze(1)     # (B, 1, 1, T)

        if device.type == "cuda":
            with autocast():
                pred = model(mix)
                loss, time_loss, spec_loss = htdemucs_style_loss(pred, target)
        else:
            pred = model(mix)
            loss, time_loss, spec_loss = htdemucs_style_loss(pred, target)

        running_loss += loss.item()
        running_time_loss += float(time_loss)
        running_spec_loss += float(spec_loss)
        num_batches += 1

        avg_loss = running_loss / num_batches
        avg_time = running_time_loss / num_batches
        avg_spec = running_spec_loss / num_batches
        elapsed = time.time() - start_time

        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            time=f"{avg_time:.4f}",
            spec=f"{avg_spec:.4f}",
            sec=f"{elapsed:.0f}",
        )

    epoch_loss = running_loss / max(num_batches, 1)
    elapsed = time.time() - start_time
    print(
        f"[Val]   Epoch {epoch} | Loss: {epoch_loss:.6f} | Time: {elapsed:.1f}s"
    )
    return epoch_loss


# ===========================
# main
# ===========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # --- DataLoader 생성 ---
    print("[Data] Building dataloaders...")
    train_loader, val_loader = create_dataloaders()

    # --- 모델 생성 ---
    # 우리 설정: mono 입력, piano 하나만 분리 → sources=["piano"], audio_channels=1
    model = HTDemucs(
        sources=["piano"],
        audio_channels=1,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] HTDemucs params: {num_params / 1e6:.2f} M")

    # --- Optimizer / AMP scaler ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=1e-4,
        min_lr=1e-6,
        verbose=True,
    )

    scaler = GradScaler(enabled=USE_AMP and device.type == "cuda")

    # --- 이어학습 로드 ---
    start_epoch = 1
    best_val_loss = float("inf")
    if RESUME_TRAIN:
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, scaler, CKPT_DIR, device
        )

    # --- 학습 루프 ---
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            use_amp=USE_AMP,
        )
        print(f"[Train] Epoch {epoch} | Loss: {train_loss:.6f}")

        # 검증
        if USE_VALIDATION and val_loader is not None:
            val_loss = validate_one_epoch(
                model,
                val_loader,
                device,
                epoch,
            )
        else:
            val_loss = train_loss  # val 안 쓸 때는 그냥 train loss 기준으로 best 관리

        scheduler.step(val_loss)
        
        # 체크포인트 저장
        last_path = save_checkpoint(
            epoch, model, optimizer, scaler, best_val_loss, CKPT_DIR
        )

        # best 갱신 시 따로 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = CKPT_DIR / "best.pth"
            torch.save(torch.load(last_path, map_location="cpu"), best_path)
            print(f"[Checkpoint] New BEST (val_loss={best_val_loss:.6f}) → {best_path}")
            
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
