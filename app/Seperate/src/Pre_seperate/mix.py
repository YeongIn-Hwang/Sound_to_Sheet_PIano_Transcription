import os
import glob
import numpy as np
import soundfile as sf
from mido import MidiFile, merge_tracks

# =========================================
# 설정: A 폴더 경로
# =========================================
ROOT = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Dataset\slakh2100\test_piano"  # <- 여기만 바꿔줘


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def mix_flac_stems(stems_dir, out_path):
    """
    stems_dir 내부의 .flac 파일들을 전부 불러와서
    길이 맞춘 뒤 합쳐서 out_path(flac)로 저장
    """
    flac_files = sorted(glob.glob(os.path.join(stems_dir, "*.flac")))
    if not flac_files:
        print(f"  [FLAC] no .flac in {stems_dir}")
        return

    audio_list = []
    sr_ref = None
    max_len = 0
    n_channels = None

    for fp in flac_files:
        try:
            audio, sr = sf.read(fp)
        except Exception as e:
            print(f"  [ERR] read {fp}: {e}")
            continue

        if sr_ref is None:
            sr_ref = sr
            n_channels = 1 if audio.ndim == 1 else audio.shape[1]
        else:
            if sr != sr_ref:
                print(f"  [WARN] sample rate mismatch: {fp} (skip)")
                continue

        # 모노/스테레오 통일
        if audio.ndim == 1 and n_channels == 2:
            # 다른 것들은 스테레오인데 이건 모노면, 스테레오로 복제
            audio = np.stack([audio, audio], axis=1)
        elif audio.ndim == 2 and n_channels == 1:
            # 다른 것들이 모노인데 이건 스테레오면, 평균해서 모노
            audio = audio.mean(axis=1)

        length = audio.shape[0]
        max_len = max(max_len, length)
        audio_list.append(audio)

    if not audio_list or sr_ref is None:
        print(f"  [FLAC] nothing mixed for {stems_dir}")
        return

    # 길이 맞추고 합치기
    stacked = []
    for a in audio_list:
        length = a.shape[0]
        if length < max_len:
            if a.ndim == 1:
                pad = np.zeros(max_len - length, dtype=a.dtype)
            else:
                pad = np.zeros((max_len - length, a.shape[1]), dtype=a.dtype)
            a = np.concatenate([a, pad], axis=0)
        stacked.append(a)

    mix = np.sum(np.stack(stacked, axis=0), axis=0)

    # 간단 normalization
    peak = np.max(np.abs(mix))
    if peak > 1e-6:
        mix = mix / peak * 0.9

    sf.write(out_path, mix, sr_ref)
    print(f"  [FLAC] saved mix -> {out_path} (sr={sr_ref}, stems={len(audio_list)})")


def merge_midi_files(midi_dir, out_path):
    """
    midi_dir 내부의 .mid 파일들을 전부 읽어서 하나로 merge
    piano_merged.mid로 저장
    """
    midi_files = sorted(glob.glob(os.path.join(midi_dir, "*.mid*")))
    if not midi_files:
        print(f"  [MIDI] no .mid in {midi_dir}")
        return

    midis = []
    ticks_per_beat = None

    for fp in midi_files:
        try:
            m = MidiFile(fp)
        except Exception as e:
            print(f"  [ERR] read MIDI {fp}: {e}")
            continue

        if ticks_per_beat is None:
            ticks_per_beat = m.ticks_per_beat
        elif ticks_per_beat != m.ticks_per_beat:
            print(f"  [WARN] ticks_per_beat mismatch in {fp} (skip)")
            continue

        midis.append(m)

    if not midis:
        print(f"  [MIDI] nothing merged for {midi_dir}")
        return

    # 모든 트랙을 하나로 합치기
    all_tracks = []
    for m in midis:
        all_tracks.extend(m.tracks)

    merged_track = merge_tracks(all_tracks)

    out_mid = MidiFile(ticks_per_beat=ticks_per_beat or 480)
    out_mid.tracks.append(merged_track)
    out_mid.save(out_path)
    print(f"  [MIDI] saved merged -> {out_path} (files={len(midis)})")


def process_track_dir(track_dir):
    """
    TrackXXXXX 폴더 하나에 대해:
      - stems/*.flac -> piano_mix.flac
      - MIDI/*.mid  -> piano_merged.mid
    """
    track_name = os.path.basename(track_dir)
    stems_dir = os.path.join(track_dir, "stems")
    midi_dir = os.path.join(track_dir, "MIDI")

    print(f"[TRACK] {track_name}")

    # stems → piano_mix.flac
    if os.path.isdir(stems_dir):
        out_flac = os.path.join(track_dir, "piano_mix.flac")
        mix_flac_stems(stems_dir, out_flac)
    else:
        print("  [FLAC] stems dir not found")

    # MIDI → piano_merged.mid
    if os.path.isdir(midi_dir):
        out_mid = os.path.join(track_dir, "piano_merged.mid")
        merge_midi_files(midi_dir, out_mid)
    else:
        print("  [MIDI] MIDI dir not found")


def main():
    for name in os.listdir(ROOT):
        track_dir = os.path.join(ROOT, name)
        if not os.path.isdir(track_dir):
            continue
        # Track으로 시작하는 폴더만 처리하고 싶으면:
        # if not name.lower().startswith("track"):
        #     continue
        process_track_dir(track_dir)


if __name__ == "__main__":
    main()
