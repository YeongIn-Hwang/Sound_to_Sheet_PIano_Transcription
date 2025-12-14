import os
import math
import random
from glob import glob

import numpy as np
import torch

# ✅ SDPA 커널 선택 강제 (FlashAttention류 비활성화)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
from tqdm.auto import tqdm


# ============================================================
#                    CONFIG / HYPERPARAMS
# ============================================================

ROOT_DIR = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Dataset\lpd_5\dense_npz"

SEGMENT_LEN = 512
BATCH_SIZE = 16          # ✅ 일단 16 추천 (VRAM 괜찮으면 32)
GRAD_ACCUM = 2           # ✅ (BATCH_SIZE * GRAD_ACCUM) = 유효 배치. 16*2=32 효과
NUM_EPOCHS = 50
LR = 2e-4                # ✅ 조금 낮추고 안정적으로
WEIGHT_DECAY = 1e-2
VAL_RATIO = 0.1
MAX_FILES = None
IN_MEMORY = False

RESUME_TRAIN = True
CKPT_PATH = "accomp_hybrid_ckpt.pt"
BEST_PATH = "accomp_hybrid_best.pt"
PATIENCE_EPOCHS = 6

NUM_WORKERS = 4
PIN_MEMORY = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
#                        DATASET
# ============================================================

class MelodyAccompDataset(Dataset):
    """
    pr_bin: (T,128,C_tracks) dense pianoroll
    - melody: 각 타임스텝에서 가장 높은 pitch 1개만 멜로디로 (현 방식 유지)
    - accomp: merged - melody
    """

    def __init__(self, root_dir, segment_len=512, max_files=None, in_memory=False):
        self.segment_len = segment_len
        self.in_memory = in_memory

        self.files = sorted(glob(os.path.join(root_dir, "*.npz")))
        if max_files is not None:
            self.files = self.files[:max_files]

        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in {root_dir}")

        self.data_list = None
        if self.in_memory:
            print(f"[Dataset] Loading {len(self.files)} files into RAM ...")
            self.data_list = []
            for p in tqdm(self.files, desc="Loading npz", dynamic_ncols=True):
                d = np.load(p)
                pr_bin = d["pr_bin"].astype(np.float32)  # (T,128,C)
                self.data_list.append(pr_bin)

    def __len__(self):
        return len(self.files)

    def _load_pr(self, idx):
        if self.in_memory:
            return self.data_list[idx]
        d = np.load(self.files[idx])
        pr_bin = d["pr_bin"].astype(np.float32)
        return pr_bin

    @staticmethod
    def _split_melody_accomp(pr_bin):
        merged = pr_bin.sum(axis=2)  # (T,128)
        T, P = merged.shape

        melody = np.zeros_like(merged, dtype=np.float32)
        for t in range(T):
            pitches = np.where(merged[t] > 0)[0]
            if len(pitches) > 0:
                melody[t, pitches.max()] = 1.0

        accomp = np.clip(merged - melody, 0, 1).astype(np.float32)
        return melody, accomp

    def _random_segment(self, mel, acc):
        T = mel.shape[0]
        L = self.segment_len

        if T < L:
            pad = L - T
            mel_pad = np.zeros((L, 128), dtype=np.float32)
            acc_pad = np.zeros((L, 128), dtype=np.float32)
            mel_pad[pad:] = mel
            acc_pad[pad:] = acc
            return mel_pad, acc_pad

        if T == L:
            return mel, acc

        start = random.randint(0, T - L)
        end = start + L
        return mel[start:end], acc[start:end]

    def __getitem__(self, idx):
        pr_bin = self._load_pr(idx)  # (T,128,C)
        melody, accomp = self._split_melody_accomp(pr_bin)
        mel_seg, acc_seg = self._random_segment(melody, accomp)

        # (B,1,T,128) 들어가도록
        x = torch.from_numpy(mel_seg[None, :, :])  # (1,T,128)
        y = torch.from_numpy(acc_seg[None, :, :])  # (1,T,128)
        return x, y


# ============================================================
#                          LOSS
# ============================================================

class FocalBCEWithLogits(nn.Module):
    """
    BCEWithLogits 기반 focal loss
    - gamma=2 기본
    - alpha는 전체 스케일/가중치(너무 크게 잡지 말기)
    """
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits/targets: (B,1,T,128)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-bce)  # pt = sigmoid prob 맞춘 정도
        focal = self.alpha * ((1 - pt) ** self.gamma) * bce
        return focal.mean()


def soft_dice_loss(logits, targets, eps=1e-6):
    """
    Soft Dice (노트 희소할 때 “아예 다 0”으로 가는 붕괴 완화)
    """
    probs = torch.sigmoid(logits)
    # flatten
    probs = probs.reshape(probs.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)

    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


class ComboLoss(nn.Module):
    """
    focal + dice (가성비 좋고, 결과 "개판" 줄이는 쪽으로 체감 큼)
    """
    def __init__(self, focal_alpha=1.0, focal_gamma=2.0, dice_weight=0.25):
        super().__init__()
        self.focal = FocalBCEWithLogits(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        lf = self.focal(logits, targets)
        ld = soft_dice_loss(logits, targets)
        return lf + self.dice_weight * ld


# ============================================================
#                          MODEL
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout(p) if p > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class TemporalAttentionBottleneck(nn.Module):
    """
    (B,C,T,F)에서
    - F 평균 풀링으로 (B,C,T) 만들고
    - MHA를 시간축(T)에만 적용 (가볍게)
    - 다시 (B,C,T,1)로 확장해서 원래 feature에 residual로 더함
    """
    def __init__(self, channels, heads=4, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x):
        # x: (B,C,T,F)
        B, C, T, F = x.shape
        g = x.mean(dim=3)          # (B,C,T)
        g = g.transpose(1, 2)      # (B,T,C)

        # attn
        h = self.norm(g)
        a, _ = self.attn(h, h, h, need_weights=False)  # (B,T,C)
        g = g + a

        # ff
        h2 = self.norm(g)
        g = g + self.ff(h2)

        # broadcast back
        g = g.transpose(1, 2).unsqueeze(3)  # (B,C,T,1)
        return x + g


class AccompHybridUNet(nn.Module):
    def __init__(self, base_ch=64, in_channels=1, out_channels=1, attn_heads=4, drop=0.0):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, base_ch, p=drop)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_ch, base_ch * 2, p=drop)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, p=drop)
        self.pool3 = nn.MaxPool2d(2)  # (T,128)->(T/8,16)

        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8, p=drop)

        # ✅ Conv + Transformer(시간) 섞기
        self.temporal_attn = TemporalAttentionBottleneck(
            channels=base_ch * 8,
            heads=attn_heads,
            dropout=drop,
        )

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4, p=drop)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2, p=drop)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch, p=drop)

        self.out_conv = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        # x: (B,1,T,128)
        e1 = self.enc1(x)      # (B,ch,T,128)
        p1 = self.pool1(e1)    # (B,ch,T/2,64)

        e2 = self.enc2(p1)     # (B,2ch,T/2,64)
        p2 = self.pool2(e2)    # (B,2ch,T/4,32)

        e3 = self.enc3(p2)     # (B,4ch,T/4,32)
        p3 = self.pool3(e3)    # (B,4ch,T/8,16)

        b = self.bottleneck(p3)            # (B,8ch,T/8,16)
        b = self.temporal_attn(b)          # ✅ (시간 구조 강화)

        u3 = self.up3(b)                   # (B,4ch,T/4,32)
        u3 = torch.cat([u3, e3], dim=1)    # (B,8ch,T/4,32)
        d3 = self.dec3(u3)                 # (B,4ch,T/4,32)

        u2 = self.up2(d3)                  # (B,2ch,T/2,64)
        u2 = torch.cat([u2, e2], dim=1)    # (B,4ch,T/2,64)
        d2 = self.dec2(u2)                 # (B,2ch,T/2,64)

        u1 = self.up1(d2)                  # (B,ch,T,128)
        u1 = torch.cat([u1, e1], dim=1)    # (B,2ch,T,128)
        d1 = self.dec1(u1)                 # (B,ch,T,128)

        out = self.out_conv(d1)            # (B,1,T,128) logits
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
#                       TRAINING LOOP
# ============================================================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if device.type == "cuda":
        cudnn.benchmark = True
        print(f"[CUDA] {torch.cuda.get_device_name(0)}")

    full_dataset = MelodyAccompDataset(
        ROOT_DIR,
        segment_len=SEGMENT_LEN,
        max_files=MAX_FILES,
        in_memory=IN_MEMORY,
    )

    n_total = len(full_dataset)
    n_val = int(n_total * VAL_RATIO)
    n_train = n_total - n_val

    indices = np.arange(n_total)
    np.random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    print(f"[Data] total={n_total}, train={len(train_dataset)}, val={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=(NUM_WORKERS > 0),
    )

    # ✅ 모델: base_ch=64 + temporal attention
    model = AccompHybridUNet(
        base_ch=64,
        in_channels=1,
        out_channels=1,
        attn_heads=4,
        drop=0.05,
    ).to(device)

    print(f"[Model] trainable params: {count_params(model):,}")

    # ✅ Loss: Focal + Dice
    criterion = ComboLoss(focal_alpha=1.0, focal_gamma=2.0, dice_weight=0.25)

    # ✅ 옵티마이저/스케줄러
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # cosine schedule (epoch 기준)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.05)

    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_no_improve = 0

    if RESUME_TRAIN and os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt and use_amp and ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        print(f"[Resume] from epoch {start_epoch-1}, best_val_loss={best_val_loss:.6f}")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # -------------------------- Train --------------------------
        model.train()
        train_loss_sum = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]", dynamic_ncols=True)
        for step, (x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)  # (B,1,T,128)
            y = y.to(device, non_blocking=True)  # (B,1,T,128)

            if use_amp:
                with autocast():
                    logits = model(x)
                    loss = criterion(logits, y) / GRAD_ACCUM
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = criterion(logits, y) / GRAD_ACCUM
                loss.backward()

            if step % GRAD_ACCUM == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss_sum += (loss.item() * GRAD_ACCUM) * x.size(0)
            pbar.set_postfix({"loss": f"{(loss.item()*GRAD_ACCUM):.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        train_loss = train_loss_sum / len(train_dataset)

        # -------------------------- Val --------------------------
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]  ", dynamic_ncols=True)
            for x, y in pbar_val:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if use_amp:
                    with autocast():
                        logits = model(x)
                        loss = criterion(logits, y)
                else:
                    logits = model(x)
                    loss = criterion(logits, y)

                val_loss_sum += loss.item() * x.size(0)
                pbar_val.set_postfix({"loss": f"{loss.item():.4f}"})

        val_loss = val_loss_sum / len(val_dataset)
        scheduler.step()

        print(f"[Epoch {epoch}] train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        # ---------------------- Checkpoint ------------------------
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve,
            "config": {
                "SEGMENT_LEN": SEGMENT_LEN,
                "BATCH_SIZE": BATCH_SIZE,
                "GRAD_ACCUM": GRAD_ACCUM,
                "LR": LR,
                "base_ch": 64,
            },
        }
        torch.save(ckpt, CKPT_PATH)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_PATH)
            print(f"[Checkpoint] New BEST saved (val_loss={val_loss:.6f}) → {BEST_PATH}")
        else:
            epochs_no_improve += 1
            print(f"[EarlyStop] no improve count = {epochs_no_improve}/{PATIENCE_EPOCHS}")
            if epochs_no_improve >= PATIENCE_EPOCHS:
                print("[EarlyStop] Stop training due to no improvement.")
                break

    print("Training finished.")


if __name__ == "__main__":
    train()
