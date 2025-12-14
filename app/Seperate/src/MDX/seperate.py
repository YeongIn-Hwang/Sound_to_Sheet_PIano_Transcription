import os
import torch
import torchaudio
from types import SimpleNamespace
import yaml
import torch.nn.functional as F
from torch.amp import autocast

from model import TFC_TDF_net


def dict_to_object(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_object(v) for k, v in d.items()})
    return d


def load_config_from_yaml(config_path: str):
    print(f"[Load config yaml] {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    return cfg_dict


def load_model(ckpt_path: str, config_path: str, device: str):
    print(f"[Load ckpt] {ckpt_path}")
    state = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=True
    )

    cfg_dict = load_config_from_yaml(config_path)
    config = dict_to_object(cfg_dict)

    model = TFC_TDF_net(config, device).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("\n=== STATE_DICT CHECK ===")
    print("missing   :", len(missing))
    print("unexpected:", len(unexpected))
    print("========================\n")

    model.eval()
    return model, config


def load_audio_with_config(path: str, audio_cfg):
    print("[Load audio]", path)

    try:
        wav, sr = torchaudio.load(path)  # mp3/wav/flac 자동 처리
    except Exception as e:
        raise RuntimeError(
            f"Failed to load audio file: {path}\n"
            f"Make sure ffmpeg is installed.\n"
            f"Original error: {e}"
        )

    target_sr = getattr(audio_cfg, "sample_rate", sr)
    num_channels = getattr(audio_cfg, "num_channels", wav.shape[0])
    chunk_size = getattr(audio_cfg, "chunk_size", None)

    # 채널 맞추기
    if wav.shape[0] < num_channels:
        rep = num_channels // wav.shape[0]
        wav = wav.repeat(rep, 1)
        wav = wav[:num_channels]
    elif wav.shape[0] > num_channels:
        wav = wav[:num_channels]

    # 리샘플
    if sr != target_sr:
        print(f"[Resample] {sr} -> {target_sr}")
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    original_len = wav.shape[1]

    # chunk padding
    if chunk_size is not None:
        pad = (-wav.shape[1]) % chunk_size
        if pad > 0:
            print(f"[Pad] +{pad} samples")
            wav = F.pad(wav, (0, pad))

    wav = wav.unsqueeze(0)  # (1, C, T)
    return wav.float(), sr, original_len

def run_inference_chunked(model, audio, device, chunk_size: int):
    audio = audio.to(device)
    b, c, t = audio.shape
    assert b == 1

    assert t % chunk_size == 0
    num_chunks = t // chunk_size
    print(f"[Inference] total length={t}, chunk_size={chunk_size}, num_chunks={num_chunks}")

    out_chunks = []
    model.eval()
    with torch.no_grad():
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = audio[:, :, start:end]

            if device == "cuda":
                with autocast(device_type="cuda"):
                    out_chunk = model(chunk)
            else:
                out_chunk = model(chunk)

            out_chunks.append(out_chunk.cpu())

    out = out_chunks[0]
    for i in range(1, len(out_chunks)):
        out = torch.cat([out, out_chunks[i]], dim=-1)

    return out.squeeze()  # 보통 (num_stems, C, T) 형태 기대


def separate_audio(ckpt_path, config_path, input_audio, stem_name: str = "Instrumental"):
    """
    stem_name: config.training.instruments 안에 있는 이름 중 하나
               예) "Vocals", "Instrumental"
    """
    # 🔹 OUTPUT 자동 생성 (stem 이름까지 포함)
    input_dir = os.path.dirname(input_audio)
    input_name = os.path.splitext(os.path.basename(input_audio))[0]
    output_path = os.path.join(input_dir, f"{input_name}_MDX_{stem_name}.wav")

    print("[Auto Output]", output_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    model, config = load_model(ckpt_path, config_path, device)
    print("[✓] Model & config loaded.")

    audio_cfg = config.audio
    audio, sr, original_len = load_audio_with_config(input_audio, audio_cfg)

    chunk_size = audio_cfg.chunk_size
    out = run_inference_chunked(model, audio, device, chunk_size)

    print("[Out shape]", out.shape)

    if out.dim() == 3:
        STEM_NAMES = getattr(config.training, "instruments", ["Vocals", "Instrumental"])
        print("[Stems in model]", STEM_NAMES)

        # stem_name이 없으면 기본 0번째
        if stem_name not in STEM_NAMES:
            print(f"[Warn] stem '{stem_name}' not in {STEM_NAMES}, default -> {STEM_NAMES[0]}")
            stem_index = 0
            stem_name_used = STEM_NAMES[0]
        else:
            stem_index = STEM_NAMES.index(stem_name)
            stem_name_used = stem_name

        print(f"[Select stem] {stem_name_used} (index={stem_index})")
        out = out[stem_index]  # (C, T) 또는 (T,)

    if out.dim() == 1:
        out_to_save = out.unsqueeze(0)  # (1, T)
    elif out.dim() == 2:
        out_to_save = out
    else:
        raise ValueError(f"Unexpected output dim: {out.dim()}")

    # pad 전 길이로 자르기
    out_to_save = out_to_save[..., :original_len]

    torchaudio.save(output_path, out_to_save, sr)
    print("[✓ Saved]", output_path)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    CKPT   = r"C:\Users\hyi8402\Desktop\Sound to Sheet\app\Seperate\src\MDX\ckpt\MDX23C_D1581.ckpt"
    CONFIG = r"C:\Users\hyi8402\Desktop\Sound to Sheet\app\Seperate\src\MDX\ckpt\model_2_stem_061321.yaml"
    INPUT  = r"C:\Users\hyi8402\Desktop\MDX\y_MDX.wav"

    # 여기에 저장하고 싶은 stem들을 나열
    STEM_LIST = ["Vocals", "Instrumental"]

    for STEM_NAME in STEM_LIST:
        print(f"\n===== Separate stem: {STEM_NAME} =====")
        separate_audio(CKPT, CONFIG, INPUT, stem_name=STEM_NAME)
