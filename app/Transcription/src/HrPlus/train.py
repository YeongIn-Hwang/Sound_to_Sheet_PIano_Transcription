# train_hrplus.py
# HRPlus 학습 스크립트 (CQT + HR 라벨)
# - AMP 사용
# - cuDNN benchmark
# - 이어학습: RESUME_TRAIN = True / False

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model import HRPlus
from Data_load import HRSegmentDataset
from data_utils import collate_hrplus

# ==========================================================
# 하드코딩 설정
# ==========================================================

# ===== 데이터 경로 =====
TRAIN_LABEL_DIR = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Dataset\Transcription\2.CQT_Data\train\Label_split"
TRAIN_FEAT_DIR  = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Dataset\Transcription\2.CQT_Data\train\Data"

VAL_LABEL_DIR   = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Dataset\Transcription\2.CQT_Data\val\Label_split"
VAL_FEAT_DIR    = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Dataset\Transcription\2.CQT_Data\val\Data"

# ===== 저장 경로 =====
SAVE_DIR = r"C:\Users\hyi8402\Desktop\Sound to Sheet\app\Transcription\2.result_hrplus"
os.makedirs(SAVE_DIR, exist_ok=True)

LAST_CKPT = os.path.join(SAVE_DIR, "hrplus_last.pt")
BEST_CKPT = os.path.join(SAVE_DIR, "hrplus_best.pt")

# ===== 이어학습 여부 =====
RESUME_TRAIN = True  # True면 이어학습, False면 처음부터

# ===== 학습 하이퍼 =====
EPOCHS      = 60
BATCH_SIZE  = 8
LR          = 1e-4
NUM_WORKERS = 2
MAX_RAM_GB  = 4
THRESHOLD   = 0.5   # F1 계산 시 sigmoid threshold


# ==========================================================
# cuDNN 설정
# ==========================================================
torch.backends.cudnn.benchmark = True


# ==========================================================
# 유틸: F1 계산
# ==========================================================
def compute_f1_from_logits(logits, targets, threshold=0.5, eps=1e-8):
    """
    logits: (B, T, K)
    targets: (B, T, K)
    """
    preds = (torch.sigmoid(logits) >= threshold).float()

    preds_f   = preds.view(-1)
    targets_f = targets.view(-1)

    tp = (preds_f * targets_f).sum().item()
    fp = (preds_f * (1 - targets_f)).sum().item()
    fn = ((1 - preds_f) * targets_f).sum().item()

    prec = tp / (tp + fp + eps) if tp + fp > 0 else 0.0
    rec  = tp / (tp + fn + eps) if tp + fn > 0 else 0.0

    f1 = 2 * prec * rec / (prec + rec + eps) if prec + rec > 0 else 0.0
    return f1


# ==========================================================
# 체크포인트 저장/로드
# ==========================================================
def save_ckpt(path, model, optimizer, scaler, epoch, best_score):
    ckpt = {
        "epoch": epoch,
        "best_score": best_score,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(ckpt, path)


def load_ckpt(path, model, optimizer=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt.get("epoch", 0) + 1
    best_score = ckpt.get("best_score", -1.0)
    return start_epoch, best_score


# ==========================================================
# TRAIN
# ==========================================================
def train_one_epoch(model, loader, optimizer, scaler, device, criterion):
    """
    loader: DataLoader with collate_hrplus
            yields: feat, onset_gt, offset_gt, frame_gt, lengths, metas
    """
    model.train()
    loss_sum = 0.0
    onset_f1_sum = 0.0
    frame_f1_sum = 0.0
    offset_f1_sum = 0.0
    steps = 0

    pbar = tqdm(loader, ncols=100)

    for (feat, onset_gt, offset_gt, frame_gt, lengths, metas) in pbar:
        feat       = feat.to(device)       # (B,1,F,T)
        onset_gt   = onset_gt.to(device)   # (B,T,K)
        offset_gt  = offset_gt.to(device)  # (B,T,K)
        frame_gt   = frame_gt.to(device)   # (B,T,K)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            out = model(feat)

            onset_logits  = out["onset_logits"]   # (B,T,K)
            offset_logits = out["offset_logits"]  # (B,T,K)
            frame_logits  = out["frame_logits"]   # (B,T,K)

            loss_onset  = criterion(onset_logits, onset_gt)
            loss_offset = criterion(offset_logits, offset_gt)
            loss_frame  = criterion(frame_logits, frame_gt)

            # 필요하면 가중치 조정 가능 (예: onset 1.0, offset 0.5, frame 1.0 등)
            loss = loss_onset + loss_offset + loss_frame

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item()
        onset_f1_sum  += compute_f1_from_logits(onset_logits.detach(),  onset_gt,  THRESHOLD)
        offset_f1_sum += compute_f1_from_logits(offset_logits.detach(), offset_gt, THRESHOLD)
        frame_f1_sum  += compute_f1_from_logits(frame_logits.detach(),  frame_gt,  THRESHOLD)
        steps += 1

        pbar.set_postfix({"loss": f"{loss_sum/steps:.4f}"})

    return (
        loss_sum / steps,
        onset_f1_sum / steps,
        offset_f1_sum / steps,
        frame_f1_sum / steps,
    )


# ==========================================================
# VAL
# ==========================================================
@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    loss_sum = 0.0
    onset_f1_sum = 0.0
    offset_f1_sum = 0.0
    frame_f1_sum = 0.0
    steps = 0

    for feat, onset_gt, offset_gt, frame_gt, lengths, metas in loader:
        feat       = feat.to(device)
        onset_gt   = onset_gt.to(device)
        offset_gt  = offset_gt.to(device)
        frame_gt   = frame_gt.to(device)

        out = model(feat)

        onset_logits  = out["onset_logits"]
        offset_logits = out["offset_logits"]
        frame_logits  = out["frame_logits"]

        loss_onset  = criterion(onset_logits, onset_gt)
        loss_offset = criterion(offset_logits, offset_gt)
        loss_frame  = criterion(frame_logits, frame_gt)
        loss = loss_onset + loss_offset + loss_frame

        loss_sum += loss.item()
        onset_f1_sum  += compute_f1_from_logits(onset_logits,  onset_gt,  THRESHOLD)
        offset_f1_sum += compute_f1_from_logits(offset_logits, offset_gt, THRESHOLD)
        frame_f1_sum  += compute_f1_from_logits(frame_logits,  frame_gt,  THRESHOLD)
        steps += 1

    return (
        loss_sum / steps,
        onset_f1_sum / steps,
        offset_f1_sum / steps,
        frame_f1_sum / steps,
    )


# ==========================================================
# MAIN
# ==========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset
    train_dataset = HRSegmentDataset(TRAIN_LABEL_DIR, TRAIN_FEAT_DIR, max_ram_gb=MAX_RAM_GB)
    val_dataset   = HRSegmentDataset(VAL_LABEL_DIR,   VAL_FEAT_DIR,   max_ram_gb=MAX_RAM_GB)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_hrplus,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_hrplus,
    )

    # Model (CQT 88bin 기준)
    model = HRPlus(
        n_pitches=88,
        cqt_bins=88,
        in_channels=1,
        base_channels=64,
        gru_hidden=128,
        pool_freq=1,  # CQT에서 이미 88bin으로 맞췄으니 추가 다운샘플링 X
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.BCEWithLogitsLoss()

    # 이어학습
    start_epoch = 1
    best_score = -1.0

    if RESUME_TRAIN and os.path.exists(LAST_CKPT):
        print(f"[Resume] Loading checkpoint: {LAST_CKPT}")
        start_epoch, best_score = load_ckpt(LAST_CKPT, model, optimizer, scaler, device)
        print(f" → Start epoch: {start_epoch}, Best score: {best_score:.4f}")

    # Main training
    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_on_f1, train_off_f1, train_fr_f1 = train_one_epoch(
            model, train_loader, optimizer, scaler, device, criterion
        )

        val_loss, val_on_f1, val_off_f1, val_fr_f1 = validate(
            model, val_loader, device, criterion
        )

        # 스코어는 onset + frame 기준으로 평균 (offset은 참고용)
        score = (val_on_f1 + val_fr_f1) / 2.0
        elapsed = time.time() - t0

        print(
            f"[Epoch {epoch}] "
            f"TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
            f"OnsetF1: {val_on_f1:.4f} | OffsetF1: {val_off_f1:.4f} | FrameF1: {val_fr_f1:.4f} | "
            f"Score: {score:.4f} | Time: {elapsed:.1f}s"
        )

        # last
        save_ckpt(LAST_CKPT, model, optimizer, scaler, epoch, best_score)

        # best
        if score > best_score:
            best_score = score
            save_ckpt(BEST_CKPT, model, optimizer, scaler, epoch, best_score)
            print(f" → New BEST saved (score={best_score:.4f})")

    print("Finished training.")
    print(f"Best score = {best_score:.4f}")


if __name__ == "__main__":
    main()
