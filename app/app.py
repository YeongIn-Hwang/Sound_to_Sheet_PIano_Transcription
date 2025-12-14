# app.py
from pathlib import Path
import tempfile
import uuid
import os

import torch

from io import BytesIO
import zipfile


from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from core.transcriber import HRPlusTranscriber
from core.Voice_Extracter import VoiceExtractor, VoiceExtractorConfig
from core.Piano_Extractor import PianoExtractor, PianoExtractorConfig
from core.arrangement import AccompService, AccompServiceConfig
from utils.file_validation import (
    TRANSCRIBE_POLICY,
    MIDI_POLICY,
    validate_upload,
    read_limited,
)


from music21 import converter, stream, note, chord, clef, instrument, meter, tempo

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

BASE_DIR = Path(__file__).resolve().parent # .../app
PROJECT_ROOT = BASE_DIR.parent # .../Sound to Sheet

CKPT_PATH = str(PROJECT_ROOT / "app" / "Transcription" / "2.result_hrplus" / "hrplus_best.pt")

MDX_DIR = PROJECT_ROOT / "app" / "Seperate" / "src" / "MDX" / "ckpt"
MDX_CKPT_PATH = str(MDX_DIR / "MDX23C_D1581.ckpt")
MDX_YAML_PATH = str(MDX_DIR / "model_2_stem_061321.yaml")

HTDE_CKPT = str(PROJECT_ROOT / "app" / "Seperate" / "src" / "htde" / "best.pth")

ACCOMP_CKPT = str(PROJECT_ROOT / "app" / "arrangement" / "ckpt" /"accomp_hybrid_best.pt")

accomp_service = AccompService(
    AccompServiceConfig(
        ckpt_path=ACCOMP_CKPT,
        device=str(DEVICE),
        fs=100,
        segment_len=512,
        overlap=256,
        thresh=0.6,
        min_note_sec=0.15,
        out_velocity=80,
    )
)

# 추가: 업로드 검증/용량 제한 유틸


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

transcriber = HRPlusTranscriber(ckpt_path=CKPT_PATH, device=DEVICE)

voice_extractor = VoiceExtractor(
    VoiceExtractorConfig(
        ckpt_path=MDX_CKPT_PATH,
        yaml_path=MDX_YAML_PATH,
        device=str(DEVICE),  # cpu로 강제면 cpu, cuda 쓰려면 DEVICE를 cuda로 바꾸면 됨
    )
)

piano_extractor = PianoExtractor(
    PianoExtractorConfig(
        ckpt_path=str(HTDE_CKPT),
        target_sr=22050,
        segment_sec=6.0,
        overlap_sec=1.0,
        device=str(DEVICE),
    )
)


@app.get("/health")
def health():
    return {"ok": True, "device": str(DEVICE)}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # ✅ 확장자 + (있으면) content-type 검증
    validate_upload(file, TRANSCRIBE_POLICY)

    tmpdir = Path(tempfile.gettempdir()) / "s2s_transcribe_in"
    tmpdir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix.lower()
    in_path = tmpdir / f"input_{uuid.uuid4().hex}{suffix}"

    try:
        # ✅ 용량 제한 걸고 안전하게 읽기(초과 시 413)
        data = await read_limited(file, TRANSCRIBE_POLICY.max_bytes)
        in_path.write_bytes(data)

        midi_bytes = transcriber.transcribe_to_midi_bytes(in_path)

        return Response(
            content=midi_bytes,
            media_type="audio/midi",
            headers={
                "Content-Disposition": 'inline; filename="transcribed.mid"',
                "Cache-Control": "no-store",
            },
        )
    finally:
        if in_path.exists():
            try:
                os.remove(in_path)
            except Exception:
                pass


# -----------------------------
# MIDI -> MusicXML (Grand Staff)
# -----------------------------
def midi_to_grandstaff_score(mid_path: str, split_pitch: int = 60):
    """
    split_pitch 기준으로 오른손/왼손을 나눠 피아노 2단(Grand Staff)처럼 보이게 만든다.
    - split_pitch=60 => C4 기준
    """
    s = converter.parse(mid_path)

    # 템포/박자(가능한 범위) 유지
    ts = s.recurse().getElementsByClass(meter.TimeSignature).first()
    mm = s.recurse().getElementsByClass(tempo.MetronomeMark).first()

    right = stream.Part()
    left = stream.Part()

    right.insert(0, instrument.Piano())
    left.insert(0, instrument.Piano())
    right.insert(0, clef.TrebleClef())
    left.insert(0, clef.BassClef())

    if ts:
        right.insert(0, ts)
        left.insert(0, ts)
    if mm:
        right.insert(0, mm)
        left.insert(0, mm)

    flat = s.flat.notesAndRests

    for el in flat:
        # Rest는 일단 생략(가독성↑). 필요하면 개선 가능.
        if el.isRest:
            continue

        off = el.offset
        dur = el.duration

        if isinstance(el, note.Note):
            tgt = right if el.pitch.midi >= split_pitch else left
            n = note.Note(el.pitch)
            n.duration = dur
            tgt.insert(off, n)

        elif isinstance(el, chord.Chord):
            r_pitches = [p for p in el.pitches if p.midi >= split_pitch]
            l_pitches = [p for p in el.pitches if p.midi < split_pitch]

            if r_pitches:
                c = chord.Chord(r_pitches)
                c.duration = dur
                right.insert(off, c)
            if l_pitches:
                c = chord.Chord(l_pitches)
                c.duration = dur
                left.insert(off, c)

    score = stream.Score()
    score.insert(0, right)
    score.insert(0, left)

    # 표기 정리
    score.makeNotation(inPlace=True)
    return score


@app.post("/midi-to-musicxml")
async def midi_to_musicxml(file: UploadFile = File(...)):
    # ✅ 확장자 + (있으면) content-type 검증
    validate_upload(file, MIDI_POLICY)

    tmpdir = Path(tempfile.gettempdir()) / "s2s_midi2xml"
    tmpdir.mkdir(parents=True, exist_ok=True)

    mid_path = tmpdir / f"in_{uuid.uuid4().hex}.mid"
    xml_path = tmpdir / f"out_{uuid.uuid4().hex}.musicxml"

    try:
        # ✅ 용량 제한 걸고 안전하게 읽기(초과 시 413)
        data = await read_limited(file, MIDI_POLICY.max_bytes)
        mid_path.write_bytes(data)

        # ✅ Grand Staff로 변환
        score = midi_to_grandstaff_score(str(mid_path), split_pitch=60)
        score.write("musicxml", fp=str(xml_path))

        xml_bytes = xml_path.read_bytes()
        return Response(
            content=xml_bytes,
            media_type="application/vnd.recordare.musicxml+xml",
            headers={
                "Content-Disposition": 'inline; filename="score.musicxml"',
                "Cache-Control": "no-store",
            },
        )
    finally:
        for p in [mid_path, xml_path]:
            try:
                if p.exists():
                    os.remove(p)
            except Exception:
                pass

@app.post("/separate/instrumental")
async def separate_instrumental(file: UploadFile = File(...)):
    validate_upload(file, TRANSCRIBE_POLICY)

    tmp_in_dir = Path(tempfile.gettempdir()) / "s2s_sep_in"
    tmp_out_dir = Path(tempfile.gettempdir()) / "s2s_sep_out"
    tmp_in_dir.mkdir(parents=True, exist_ok=True)
    tmp_out_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix.lower()
    in_path = tmp_in_dir / f"in_{uuid.uuid4().hex}{suffix}"
    out_path = tmp_out_dir / f"inst_{uuid.uuid4().hex}.wav"

    try:
        data = await read_limited(file, TRANSCRIBE_POLICY.max_bytes)
        in_path.write_bytes(data)

        voice_extractor.extract_instrumental_to_wav(
            input_audio_path=str(in_path),
            output_wav_path=str(out_path),
            stem_name="Instrumental",
        )

        wav_bytes = out_path.read_bytes()
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="instrumental.wav"',
                "Cache-Control": "no-store",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [in_path, out_path]:
            try:
                if p.exists():
                    os.remove(p)
            except Exception:
                pass


@app.post("/separate/piano")
async def separate_piano(
    file: UploadFile = File(...),
    has_vocals: bool = Form(False),  # ✅ 이거 필수
):
    validate_upload(file, TRANSCRIBE_POLICY)

    tmp_in_dir = Path(tempfile.gettempdir()) / "s2s_sep_in"
    tmp_mid_dir = Path(tempfile.gettempdir()) / "s2s_sep_mid"
    tmp_out_dir = Path(tempfile.gettempdir()) / "s2s_sep_out"
    tmp_in_dir.mkdir(parents=True, exist_ok=True)
    tmp_mid_dir.mkdir(parents=True, exist_ok=True)
    tmp_out_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix.lower()
    in_path = tmp_in_dir / f"in_{uuid.uuid4().hex}{suffix}"
    inst_path = tmp_mid_dir / f"inst_{uuid.uuid4().hex}.wav"
    out_path = tmp_out_dir / f"piano_{uuid.uuid4().hex}.wav"

    try:
        data = await read_limited(file, TRANSCRIBE_POLICY.max_bytes)
        in_path.write_bytes(data)

        # ✅ 핵심 분기
        piano_input = in_path
        if has_vocals:
            print("[MDX] extracting Instrumental…")
            voice_extractor.extract_instrumental_to_wav(
                input_audio_path=str(in_path),
                output_wav_path=str(inst_path),
                stem_name="Instrumental",
            )
            piano_input = inst_path

        print("[PIANO] extracting Piano…")
        piano_extractor.extract_piano_to_wav(
            input_audio_path=str(piano_input),
            output_wav_path=str(out_path),
        )

        return Response(
            content=out_path.read_bytes(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="piano.wav"',
                "Cache-Control": "no-store",
            },
        )

    finally:
        for p in [in_path, inst_path, out_path]:
            try:
                if p.exists():
                    os.remove(p)
            except Exception:
                pass
    
@app.post("/arrangement/midi")
async def arrangement_midi(
    file: UploadFile = File(...),
    keep_original: bool = Form(True),
):
    # ✅ midi 업로드 검증
    #validate_upload(file, MIDI_POLICY)

    # ✅ 용량 제한 걸고 읽기
    midi_bytes = await read_limited(file, MIDI_POLICY.max_bytes)

    try:
        out_bytes = accomp_service.generate_midi_bytes(
            input_midi_bytes=midi_bytes,
            keep_original=keep_original,
        )
        return Response(
            content=out_bytes,
            media_type="audio/midi",
            headers={
                "Content-Disposition": 'inline; filename="accomp_pred.mid"',
                "Cache-Control": "no-store",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))