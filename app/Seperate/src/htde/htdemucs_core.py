# htdemucs_core.py
# HTDemucs에서 실제로 사용하는 공통 유틸/레이어 모음
# (capture_init, STFT/ISTFT, rescale, DConv, HEnc/HDec, MultiWrap 등)

import functools
import typing as tp

import torch
import torch as th
from torch import nn
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# 1. states.capture_init
# -----------------------------------------------------------------------------
def capture_init(init):
    """
    __init__ 인자를 self._init_args_kwargs 에 저장해 두는 데코레이터.
    serialize_model/load_model 할 때 사용.
    """
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__


# -----------------------------------------------------------------------------
# 2. spec.spectro / spec.ispectro (STFT / ISTFT)
#    - 원본 Demucs spec.py 기반
# -----------------------------------------------------------------------------
def spectro(x, n_fft: int = 512, hop_length: tp.Optional[int] = None, pad: int = 0):
    """
    x: (..., T) 또는 (B, C, T) 형태 텐서
    return: (..., F, T_spec) 복소 STFT
    """
    *other, length = x.shape
    x = x.reshape(-1, length)

    device_type = x.device.type
    is_other_gpu = device_type not in ["cuda", "cpu"]

    if is_other_gpu:
        x = x.cpu()

    z = th.stft(
        x,
        n_fft * (1 + pad),
        hop_length or n_fft // 4,
        window=th.hann_window(n_fft).to(x),
        win_length=n_fft,
        normalized=True,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )
    _, freqs, frames = z.shape
    return z.view(*other, freqs, frames)


def ispectro(
    z,
    hop_length: tp.Optional[int] = None,
    length: tp.Optional[int] = None,
    pad: int = 0,
):
    """
    z: (..., F, T_spec) 복소 STFT
    return: (..., T) 실제 파형
    """
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)

    device_type = z.device.type
    is_other_gpu = device_type not in ["cuda", "cpu"]

    if is_other_gpu:
        z = z.cpu()

    x = th.istft(
        z,
        n_fft,
        hop_length,
        window=th.hann_window(win_length).to(z.real),
        win_length=win_length,
        normalized=True,
        length=length,
        center=True,
    )
    _, length = x.shape
    return x.view(*other, length)


# -----------------------------------------------------------------------------
# 3. demucs.rescale_conv / rescale_module
#    - conv weight 초기 std를 reference에 맞추기 위해 사용
# -----------------------------------------------------------------------------
def rescale_conv(conv: nn.Module, reference: float):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module: nn.Module, reference: float):
    """
    Conv 계열 레이어들의 초기 weight scale을 통일해 주는 유틸.
    """
    for sub in module.modules():
        if isinstance(
            sub,
            (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d),
        ):
            rescale_conv(sub, reference)


# -----------------------------------------------------------------------------
# 4. Demucs DConv (minimal 버전)
#    - attn / lstm 인자는 받지만 실제로는 사용하지 않는 간소화 버전.
#    - HTDemucs 기본 설정(dconv_kw)에서는 attn/lstm=False라서 구조 동일.
# -----------------------------------------------------------------------------
class LayerScale(nn.Module):
    """
    Layer scale from [Touvron et al. 2021].
    잔차(residual)를 per-channel scale 파라미터로 스케일링.
    """

    def __init__(self, channels: int, init: float = 0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        # x: (B, C, T)
        return self.scale[:, None] * x


class DConv(nn.Module):
    """
    Demucs에서 encoder layer 안에 들어가는 residual branch.

    - (Conv1d -> Norm -> GELU -> Conv1d -> Norm -> GLU -> LayerScale) * depth
    - 입력과 같은 shape를 유지하는 residual block.
    """

    def __init__(
        self,
        channels: int,
        compress: float = 4.0,
        depth: int = 2,
        init: float = 1e-4,
        norm: bool = True,
        attn: bool = False,   # minimal core에서는 사용 안 함
        heads: int = 4,
        ndecay: int = 4,
        lstm: bool = False,   # minimal core에서는 사용 안 함
        gelu: bool = True,
        kernel: int = 3,
        dilate: bool = True,
    ):
        """
        Args:
            channels: 입력/출력 채널 수
            compress: 내부 hidden 채널 축소 비율 (hidden = channels / compress)
            depth: 몇 개의 residual 층을 쌓을지
            init: LayerScale 초기값
            norm: GroupNorm 사용할지 여부
            attn / lstm: 원본 Demucs 옵션이지만 여기서는 비활성
            gelu: 활성함수로 GELU 사용할지
            kernel: Conv1d 커널 크기 (홀수)
            dilate: depth 에 따라 dilation 1,2,4,... 사용할지
        """
        super().__init__()
        assert kernel % 2 == 1
        self.channels = channels
        self.compress = compress
        self.depth = abs(depth)

        # attention / lstm은 최소 구현에서는 지원 X
        if attn or lstm:
            raise NotImplementedError(
                "Minimal DConv in htdemucs_core does not support attn/lstm. "
                "Use default HTDemucs settings (attn=False, lstm=False)."
            )

        # 노말라이제이션
        norm_fn: tp.Callable[[int], nn.Module] = lambda d: nn.Identity()
        if norm:
            norm_fn = lambda d: nn.GroupNorm(1, d)

        hidden = int(channels / compress)

        act: tp.Type[nn.Module]
        act = nn.GELU if gelu else nn.ReLU

        self.layers = nn.ModuleList([])
        for d in range(self.depth):
            dilation = 2 ** d if dilate else 1
            padding = dilation * (kernel // 2)

            mods = [
                nn.Conv1d(
                    channels,
                    hidden,
                    kernel,
                    dilation=dilation,
                    padding=padding,
                ),
                norm_fn(hidden),
                act(),
                nn.Conv1d(hidden, 2 * channels, 1),
                norm_fn(2 * channels),
                nn.GLU(dim=1),
                LayerScale(channels, init),
            ]
            self.layers.append(nn.Sequential(*mods))

    def forward(self, x):
        """
        x: (B, C, T)
        """
        for layer in self.layers:
            x = x + layer(x)
        return x


# -----------------------------------------------------------------------------
# 5. hdemucs.pad1d / ScaledEmbedding / HEncLayer / HDecLayer / MultiWrap
#    - 원본 Demucs hdemucs.py 기반 (DConv만 위에서 재구현)
# -----------------------------------------------------------------------------
def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
):
    """
    F.pad 의 thin wrapper.

    reflect padding에서 입력 길이가 짧을 때 에러를 피하기 위해
    한 번 더 오른쪽에 0-padding을 추가한 뒤 reflect 를 적용하는 트릭을 사용.
    """
    x0 = x
    length = x.shape[-1]
    padding_left, padding_right = paddings
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            extra_pad_right = min(padding_right, extra_pad)
            extra_pad_left = extra_pad - extra_pad_right
            paddings = (
                padding_left - extra_pad_left,
                padding_right - extra_pad_right,
            )
            x = F.pad(x, (extra_pad_left, extra_pad_right))
    out = F.pad(x, paddings, mode, value)
    assert out.shape[-1] == length + padding_left + padding_right
    assert (out[..., padding_left: padding_left + length] == x0).all()
    return out


class ScaledEmbedding(nn.Module):
    """
    Embedding 계층에 대해 learning rate를 키우는 효과 (scale) + smooth 초기화 옵션.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        scale: float = 10.0,
        smooth: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:
            weight = torch.cumsum(self.embedding.weight.data, dim=0)
            # gaussian 합의 분산 증가를 고려하여 sqrt(n)으로 나눔
            weight = weight / torch.arange(
                1, num_embeddings + 1, device=weight.device
            ).sqrt()[:, None]
            self.embedding.weight.data[:] = weight
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def forward(self, x):
        return self.embedding(x) * self.scale


class HEncLayer(nn.Module):
    """
    Hybrid Encoder layer.
    - 시간 도메인 / 주파수 도메인 모두에서 사용 가능 (freq 플래그로 구분)
    """

    def __init__(
        self,
        chin,
        chout,
        kernel_size=8,
        stride=4,
        norm_groups=1,
        empty=False,
        freq=True,
        dconv=True,
        norm=True,
        context=0,
        dconv_kw=None,
        pad=True,
        rewrite=True,
    ):
        """
        Args:
            chin: 입력 채널 수
            chout: 출력 채널 수
            norm_groups: GroupNorm 그룹 수
            empty: conv만 하고 rewrite/dconv를 생략할지 여부
            freq: 주파수 축 conv를 할지 여부 (True면 Conv2d)
            dconv: DConv residual 사용 여부
            norm: GroupNorm 사용 여부
            context: 1x1 conv의 양쪽 context 크기
            dconv_kw: DConv 에 들어갈 kwargs dict
            pad: stride에 맞춰 항상 길이가 나누어 떨어지도록 pad
            rewrite: 마지막에 1x1 conv + GLU로 다시 쓰기(rewrite) 할지
        """
        super().__init__()
        if dconv_kw is None:
            dconv_kw = {}

        norm_fn = lambda d: nn.Identity()
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)

        if pad:
            pad_val = kernel_size // 4
        else:
            pad_val = 0

        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.norm = norm

        klass = nn.Conv1d
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            pad_val = [pad_val, 0]
            klass = nn.Conv2d

        self.conv = klass(chin, chout, kernel_size, stride, pad_val)
        if self.empty:
            return

        self.norm1 = norm_fn(chout)
        self.rewrite = None
        if rewrite:
            self.rewrite = klass(
                chout,
                2 * chout,
                1 + 2 * context,
                1,
                context,
            )
            self.norm2 = norm_fn(2 * chout)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chout, **dconv_kw)

    def forward(self, x, inject=None):
        """
        `inject`: time branch 결과를 freq branch에 주입할 때 사용 (stride 맞을 때).
        """
        if not self.freq and x.dim() == 4:
            # freq branch에서 time branch로 변환
            B, C, Fr, T = x.shape
            x = x.view(B, -1, T)

        if not self.freq:
            length = x.shape[-1]
            if length % self.stride != 0:
                x = F.pad(x, (0, self.stride - (length % self.stride)))

        y = self.conv(x)
        if self.empty:
            return y

        if inject is not None:
            assert inject.shape[-1] == y.shape[-1], (inject.shape, y.shape)
            if inject.dim() == 3 and y.dim() == 4:
                inject = inject[:, :, None]
            y = y + inject

        y = F.gelu(self.norm1(y))

        if self.dconv is not None:
            if self.freq:
                B, C, Fr, T = y.shape
                y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv(y)
            if self.freq:
                y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)

        if self.rewrite is not None:
            z = self.norm2(self.rewrite(y))
            z = F.glu(z, dim=1)
        else:
            z = y
        return z


class HDecLayer(nn.Module):
    """
    Hybrid Decoder layer.
    Encoder와 비슷하지만 ConvTranspose를 사용하고 skip connection을 받음.
    """

    def __init__(
        self,
        chin,
        chout,
        last=False,
        kernel_size=8,
        stride=4,
        norm_groups=1,
        empty=False,
        freq=True,
        dconv=True,
        norm=True,
        context=1,
        dconv_kw=None,
        pad=True,
        context_freq=True,
        rewrite=True,
    ):
        """
        See HEncLayer for docs.
        """
        super().__init__()
        if dconv_kw is None:
            dconv_kw = {}

        norm_fn = lambda d: nn.Identity()
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)

        if pad:
            pad_val = kernel_size // 4
        else:
            pad_val = 0
        self.pad = pad_val
        self.last = last

        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        self.norm = norm
        self.context_freq = context_freq
        self.freq = freq

        klass = nn.Conv1d
        klass_tr = nn.ConvTranspose1d
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            klass = nn.Conv2d
            klass_tr = nn.ConvTranspose2d

        self.conv_tr = klass_tr(chin, chout, kernel_size, stride)
        self.norm2 = norm_fn(chout)

        if self.empty:
            return

        self.rewrite = None
        if rewrite:
            if context_freq:
                self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context)
            else:
                self.rewrite = klass(
                    chin,
                    2 * chin,
                    [1, 1 + 2 * context],
                    1,
                    [0, context],
                )
            self.norm1 = norm_fn(2 * chin)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chin, **dconv_kw)

    def forward(self, x, skip=None, length=None):
        """
        x: 입력 feature
        skip: encoder에서 넘어온 skip connection
        length: time branch에서 crop할 길이
        """
        if not self.freq and x.dim() == 4:
            B, C, Fr, T = x.shape
            x = x.view(B, -1, T)

        if skip is not None:
            assert skip.shape[-1] == x.shape[-1], (skip.shape, x.shape)

        if self.empty:
            y = x
        else:
            if self.rewrite is not None:
                y = F.glu(self.norm1(self.rewrite(x)), dim=1)
            else:
                y = x

            if self.dconv is not None:
                if self.freq:
                    B, C, Fr, T = y.shape
                    y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
                y = self.dconv(y)
                if self.freq:
                    y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)

        # skip 추가
        if skip is not None:
            y = y + skip

        z = self.norm2(self.conv_tr(y))
        if self.freq:
            if self.pad:
                # freq branch: (B, C, F, T) 에서 F축을 crop
                z = z[:, :, self.pad:-self.pad, :]
        else:
            assert length is not None
            # time branch: 마지막 축(T) crop
            z = z[..., self.pad:self.pad + length]
            assert z.shape[-1] == length, (z.shape[-1], length)

        if not self.last:
            z = F.gelu(z)
        return z, y


class MultiWrap(nn.Module):
    """
    HEncLayer / HDecLayer 를 여러 주파수 밴드로 쪼개서 적용하는 wrapper.

    - split_ratios: 각 밴드가 전체 주파수 중 차지하는 비율.
    """

    def __init__(self, layer: nn.Module, split_ratios: tp.List[float]):
        """
        Args:
            layer: HEncLayer 또는 HDecLayer 인스턴스
            split_ratios: 각 band의 주파수 비율 리스트
        """
        super().__init__()
        assert isinstance(layer, (HEncLayer, HDecLayer))
        self.split_ratios = split_ratios

        # replica들을 모아 놓는다.
        self.layers = nn.ModuleList()
        for _ in split_ratios:
            # 여기서는 구조 단순화를 위해 layer의 non-module 속성들만 복사하지 않고,
            # 동일 클래스를 기본 인자로 다시 생성하는 대신,
            # 그대로 deepcopy를 쓰는 원본 구현 대신 단순 래핑만 한다.
            # (기본 HTDemucs 설정에서는 multi_freqs=None 이라 MultiWrap을 사용하지 않음)
            self.layers.append(layer)

        # 마지막에 conv만 있는 간단한 wrapper인지 (HEnc empty-layer 형태) 여부
        self.conv = isinstance(layer, HEncLayer) and layer.empty

    def forward(self, x, *extra):
        """
        x: (B, C, F, T) 형태 (freq branch 기준)
        extra: HEncLayer/HDecLayer에서 추가로 받는 인자 (inject, skip 등)
        """
        # 기본 설정에서는 multi_freqs=None 이라서 MultiWrap이 호출되지 않지만,
        # 혹시라도 사용될 경우를 위해 매우 단순하게 한 번만 통과시키는 버전.
        layer = self.layers[0]

        if self.conv:
            out = layer.conv(x)
            return out
        else:
            out = layer(x, *extra)
            return out
