# model.py
import torch
import torch.nn as nn

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
    def __init__(self, channels, heads=4, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x):
        g = x.mean(dim=3).transpose(1, 2)  # (B,T,C)
        h = self.norm(g)
        a, _ = self.attn(h, h, h, need_weights=False)
        g = g + a
        g = g + self.ff(self.norm(g))
        return x + g.transpose(1, 2).unsqueeze(3)


class AccompHybridUNet(nn.Module):
    def __init__(self, base_ch=64, attn_heads=4, drop=0.05):
        super().__init__()

        self.enc1 = ConvBlock(1, base_ch, p=drop)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_ch, base_ch * 2, p=drop)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, p=drop)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8, p=drop)
        self.temporal_attn = TemporalAttentionBottleneck(
            base_ch * 8, heads=attn_heads, dropout=drop
        )

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4, p=drop)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2, p=drop)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch, p=drop)

        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)

        b = self.temporal_attn(self.bottleneck(p3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)
