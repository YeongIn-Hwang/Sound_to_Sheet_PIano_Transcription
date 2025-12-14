# model_hrplus.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConvBlock(nn.Module):
    """
    HRplus 논문 Fig.2(b) 스타일의 Dilated Conv Block 간단 구현.
    - 입력: (B, C_in, F, T)  [F: frequency bins, T: time frames]
    - 출력: (B, C_out, F_reduced, T)
    """
    def __init__(self, in_channels=1, base_channels=64, pool_freq=4):
        super().__init__()
        self.pool_freq = pool_freq

        # 1) 7x7 Conv x3 + InstNorm + ReLU
        convs_7x7 = []
        c_in = in_channels
        for _ in range(3):
            convs_7x7.append(
                nn.Sequential(
                    nn.Conv2d(c_in, base_channels, kernel_size=7, padding=3),
                    nn.InstanceNorm2d(base_channels, affine=True),
                    nn.ReLU(inplace=True),
                )
            )
            c_in = base_channels
        self.convs_7x7 = nn.ModuleList(convs_7x7)

        # 2) Dilated conv 1x3 x 8 (시간축에 dilation 적용)
        # 논문에서는 dilation=48,76,... 이런 식으로 쓰지만,
        # 여기서는 그대로 숫자만 맞춰줌.
        dil_rates = [48, 76, 96, 111, 124, 135, 144, 152]
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    base_channels,
                    base_channels,
                    kernel_size=(1, 3),
                    padding=(0, d),
                    dilation=(1, d),
                ),
                nn.InstanceNorm2d(base_channels, affine=True),
                nn.ReLU(inplace=True),
            )
            for d in dil_rates
        ])

        # 8개 dilated conv 출력 합친 뒤 1x3 dilated conv
        self.merge_dilated = nn.Sequential(
            nn.Conv2d(
                base_channels * len(dil_rates),
                base_channels,
                kernel_size=(1, 3),
                padding=(0, 48),
                dilation=(1, 48),
            ),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
        )

        # MaxPool(freq) → 피아노 키 개수로 줄이기 용
        # (freq / pool_freq)로 줄어듦. CQT를 352bin 쓰고 pool=4면 88bin.
        self.freq_pool = nn.MaxPool2d(kernel_size=(pool_freq, 1))

        # 마지막 정리용 conv 들 (5x1 비슷하게)
        self.post_convs = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=(1, 3), padding=(0, 12), dilation=(1, 12)),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=(5, 1), padding=(2, 0)),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=(5, 1), padding=(2, 0)),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        x: (B, 1, F, T)  - CQT 입력
        return: (B, C, F', T)
        """
        h = x
        for conv in self.convs_7x7:
            h = conv(h)

        dil_outputs = []
        for dc in self.dilated_convs:
            dil_outputs.append(dc(h))
        h = torch.cat(dil_outputs, dim=1)  # (B, C*8, F, T)

        h = self.merge_dilated(h)         # (B, C, F, T)
        h = self.freq_pool(h)             # (B, C, F', T)  (F'≈88)
        h = self.post_convs(h)            # (B, C, F', T)
        return h


class HRPlus(nn.Module):
    """
    HRplus 스타일의 멀티헤드 모델 (Velocity / Onset / Offset / Frame)
    - 입력:  x  : (B, 1, F, T)  CQT
    - 출력: dict:
        {
          "velocity": (B, T, K),
          "onset":    (B, T, K),
          "offset":   (B, T, K),
          "frame":    (B, T, K),
        }
      (K: 피치 개수, 보통 88)
    """
    def __init__(
        self,
        n_pitches: int = 88,
        cqt_bins: int = 352,       # CQT freq bin 수 (352 쓰고 pool=4 → 88)
        in_channels: int = 1,
        base_channels: int = 64,
        gru_hidden: int = 128,
        pool_freq: int = 4,
    ):
        super().__init__()
        self.n_pitches = n_pitches
        self.cqt_bins = cqt_bins

        # 1) CQT → Acoustic feature (CRNN front)
        self.acoustic_cnn = DilatedConvBlock(
            in_channels=in_channels,
            base_channels=base_channels,
            pool_freq=pool_freq,
        )

        # freq축을 pitch 축으로 매핑하는 1x1 conv (optional)
        # (B, C, F', T) → (B, C, K, T)
        self.freq_to_pitch = nn.Conv2d(
            base_channels,
            base_channels,
            kernel_size=(1, 1)
        )

        # 시간축 RNN (shared acoustic GRU)
        # 입력: 각 프레임당 feature = C * K (flatten)
        self.gru_input_dim = base_channels * self.n_pitches
        self.acoustic_gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        acoustic_feat_dim = gru_hidden * 2  # BiGRU

        # ============== 각 task head ==============

        # Velocity head: acoustic_feat → vel (B,T,K)
        self.vel_fc = nn.Linear(acoustic_feat_dim, self.n_pitches)

        # Onset head: [acoustic_feat, vel_pred] → onset
        self.onset_gru = nn.GRU(
            input_size=acoustic_feat_dim + self.n_pitches,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.onset_fc = nn.Linear(gru_hidden * 2, self.n_pitches)

        # Offset head: [acoustic_feat] → offset
        self.offset_gru = nn.GRU(
            input_size=acoustic_feat_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.offset_fc = nn.Linear(gru_hidden * 2, self.n_pitches)

        # Frame head: [acoustic_feat, onset_pred, offset_pred] → frame
        self.frame_gru = nn.GRU(
            input_size=acoustic_feat_dim + self.n_pitches * 2,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.frame_fc = nn.Linear(gru_hidden * 2, self.n_pitches)

    def forward(self, x):
        """
        x: (B, 1, F, T)  ㅡ CQT 입력 (nnAudio 등으로 만든 것)
        """
        B, C, F, T = x.shape

        # 1) Acoustic CNN
        h = self.acoustic_cnn(x)  # (B, C', F', T)

        # 2) freq → pitch 축 정리
        #   여기서는 단순히 conv 후, freq축을 n_pitches로 맞춘다고 가정
        #   실제로는 F'가 n_pitches와 같도록 CQT+pooling 설정을 맞추는게 깔끔함.
        h = self.freq_to_pitch(h)         # (B, C', F', T)
        # 만약 F' != n_pitches라면, interpolation으로 맞춰준다
        if h.shape[2] != self.n_pitches:
            h = F.interpolate(
                h,
                size=(self.n_pitches, T),
                mode="bilinear",
                align_corners=False,
            )  # (B, C', K, T)

        # 3) (B, C', K, T) → (B, T, C'*K) 로 바꿔서 GRU 입력
        h = h.permute(0, 3, 1, 2).contiguous()   # (B, T, C', K)
        B, T, Cc, K = h.shape
        h_flat = h.view(B, T, Cc * K)            # (B, T, C'*K)

        # 4) Acoustic GRU
        acoustic_feat, _ = self.acoustic_gru(h_flat)  # (B, T, H*2)

        # ========== Velocity ==========
        vel_logits = self.vel_fc(acoustic_feat)       # (B, T, K)
        vel_pred = torch.sigmoid(vel_logits)

        # ========== Onset ==========
        onset_in = torch.cat([acoustic_feat, vel_pred], dim=-1)  # (B,T,H*2+K)
        onset_feat, _ = self.onset_gru(onset_in)
        onset_logits = self.onset_fc(onset_feat)
        onset_pred = torch.sigmoid(onset_logits)

        # ========== Offset ==========
        offset_feat, _ = self.offset_gru(acoustic_feat)
        offset_logits = self.offset_fc(offset_feat)
        offset_pred = torch.sigmoid(offset_logits)

        # ========== Frame ==========
        frame_in = torch.cat([acoustic_feat, onset_pred, offset_pred], dim=-1)
        frame_feat, _ = self.frame_gru(frame_in)
        frame_logits = self.frame_fc(frame_feat)
        frame_pred = torch.sigmoid(frame_logits)

        return {
            "velocity": vel_pred,
            "onset": onset_pred,
            "offset": offset_pred,
            "frame": frame_pred,
            # 필요하면 logits도 같이 반환해도 됨
            "velocity_logits": vel_logits,
            "onset_logits": onset_logits,
            "offset_logits": offset_logits,
            "frame_logits": frame_logits,
        }
