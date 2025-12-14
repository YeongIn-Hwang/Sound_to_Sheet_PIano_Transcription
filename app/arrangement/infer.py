import numpy as np
import torch
import torch.nn as nn

from model import AccompHybridUNet

# =======================
# HARD-CODED PATHS
# =======================
INPUT_MID = r"C:\Users\hyi8402\Desktop\비행기_hrplus.mid"
CKPT_PATH = r"C:\Users\hyi8402\Desktop\Sound to Sheet\app\arrangement\ckpt\accomp_hybrid_best.pt"
OUT_MIDI  = r"C:\Users\hyi8402\Desktop\Sound to Sheet\app\arrangement\output.mid"

# =======================
# SETTINGS
# =======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEGMENT_LEN = 512
OVERLAP = 256

FS = 100              # 🎯 학습 데이터와 동일 (LPD-5 계열)
THRESH = 0.6          # 살짝 높이는 게 보통 깔끔
MIN_NOTE_SEC = 0.15   # 짧은 잡음 제거

SAVE_ORIGINAL = True

# =======================
# MIDI → Melody roll
# =======================
def midi_to_melody_roll(midi_path, fs=FS):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    pr = pm.get_piano_roll(fs=fs)      # (128, T)
    pr = (pr > 0).astype(np.float32).T # (T,128)

    melody = np.zeros_like(pr)
    for t in range(pr.shape[0]):
        pitches = np.where(pr[t] > 0)[0]
        if len(pitches) > 0:
            melody[t, pitches.max()] = 1.0

    return melody, 1.0 / fs

# =======================
# Sliding-window inference
# =======================
def infer_full_song(model, melody_roll):
    T = melody_roll.shape[0]
    probs_sum = np.zeros((T, 128), dtype=np.float32)
    weight = np.zeros((T, 128), dtype=np.float32)

    step = max(1, SEGMENT_LEN - OVERLAP)

    model.eval()
    with torch.no_grad():
        for start in range(0, T, step):
            seg = melody_roll[start:start + SEGMENT_LEN]
            if seg.shape[0] < SEGMENT_LEN:
                seg = np.pad(seg, ((0, SEGMENT_LEN - seg.shape[0]), (0, 0)))

            x = torch.from_numpy(seg[None, None]).to(DEVICE)
            p = torch.sigmoid(model(x))[0, 0].cpu().numpy()

            valid = min(SEGMENT_LEN, T - start)
            probs_sum[start:start + valid] += p[:valid]
            weight[start:start + valid] += 1.0

    return probs_sum / np.maximum(weight, 1e-6)

# =======================
# Roll → MIDI
# =======================
def roll_to_midi(binary_roll, frame_sec, out_midi_path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)

    T, P = binary_roll.shape
    min_frames = max(1, int(MIN_NOTE_SEC / frame_sec))

    for pitch in range(P):
        on = None
        run = 0
        for t in range(T):
            if binary_roll[t, pitch]:
                if on is None:
                    on = t; run = 1
                else:
                    run += 1
            elif on is not None:
                if run >= min_frames:
                    inst.notes.append(pretty_midi.Note(
                        velocity=90, pitch=pitch,
                        start=on * frame_sec, end=t * frame_sec
                    ))
                on = None; run = 0

        if on is not None and run >= min_frames:
            inst.notes.append(pretty_midi.Note(
                velocity=90, pitch=pitch,
                start=on * frame_sec, end=T * frame_sec
            ))

    pm.instruments.append(inst)
    pm.write(out_midi_path)

def roll_to_instrument(binary_roll, frame_sec, velocity=90):
    import pretty_midi
    inst = pretty_midi.Instrument(program=0)  # Piano

    T, P = binary_roll.shape
    min_frames = max(1, int(MIN_NOTE_SEC / frame_sec))

    for pitch in range(P):
        on = None
        run = 0
        for t in range(T):
            if binary_roll[t, pitch]:
                if on is None:
                    on = t; run = 1
                else:
                    run += 1
            elif on is not None:
                if run >= min_frames:
                    inst.notes.append(pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=on * frame_sec,
                        end=t * frame_sec
                    ))
                on = None; run = 0

        if on is not None and run >= min_frames:
            inst.notes.append(pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=on * frame_sec,
                end=T * frame_sec
            ))
    return inst

# =======================
# MAIN
# =======================
def main():
    import pretty_midi

    print(f"[Device] {DEVICE}")
    print(f"[Input MIDI] {INPUT_MID}")

    # 1) 멜로디 roll 생성 (모델 입력용)
    melody_roll, frame_sec = midi_to_melody_roll(INPUT_MID)
    print(f"[Melody roll] {melody_roll.shape}, frame_sec={frame_sec:.6f}")

    # 2) 모델 로드
    model = AccompHybridUNet().to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)

    # 3) 반주 생성
    probs = infer_full_song(model, melody_roll)
    binary = (probs >= THRESH).astype(np.uint8)

    # 4) 빈 MIDI에 반주 트랙만 추가
    pm_out = pretty_midi.PrettyMIDI(INPUT_MID) if SAVE_ORIGINAL else pretty_midi.PrettyMIDI()
    accomp_inst = roll_to_instrument(binary, frame_sec, velocity=80)
    pm_out.instruments.append(accomp_inst)

    # 5) 저장
    pm_out.write(OUT_MIDI)
    print(f"[Saved] {OUT_MIDI}")

if __name__ == "__main__":
    main()
