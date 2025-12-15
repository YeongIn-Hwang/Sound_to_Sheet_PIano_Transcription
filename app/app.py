# app.py
from pathlib import Path
import tempfile
import uuid
import os

import torch

from io import BytesIO
import zipfile

# =========================
# DEBUG IMPORTS (added)
# =========================
import time
import threading
import faulthandler
import signal
import sys
import traceback

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

# =========================
# DEBUG UTILS (added)
# =========================
DEBUG_BOOT = os.getenv("S2S_DEBUG_BOOT", "1") == "1"
DEBUG_REQUESTS = os.getenv("S2S_DEBUG_REQUESTS", "1") == "1"
DUMP_INTERVAL_SEC = int(os.getenv("S2S_DUMP_INTERVAL_SEC", "20"))

def _dbg(msg: str):
    if DEBUG_BOOT:
        print(f"[BOOT][{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def _print_path_info(label: str, path_str: str):
    try:
        p = Path(path_str)
        exists = p.exists()
        size = p.stat().st_size if exists else None
        _dbg(f"{label}: {path_str}")
        _dbg(f"  -> exists={exists}, size={size}, is_file={p.is_file() if exists else None}")
    except Exception as e:
        _dbg(f"{label}: {path_str} (path info error: {e})")

def _start_periodic_trace_dump():
    """
    If the process 'hangs' (e.g., torch.load stuck), periodically dump stack traces.
    """
    if not DEBUG_BOOT:
        return

    faulthandler.enable(all_threads=True)

    def _loop():
        while True:
            time.sleep(DUMP_INTERVAL_SEC)
            _dbg(f"Periodic trace dump (every {DUMP_INTERVAL_SEC}s)")
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    _dbg("Started periodic traceback dumper thread")

def _install_signal_dump():
    """
    Send SIGUSR1 to dump stack trace (Linux). On Windows this won't work, but RunPod is Linux.
    """
    if not DEBUG_BOOT:
        return

    def _handler(signum, frame):
        _dbg(f"Signal {signum} received -> dumping traceback")
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _handler)
        _dbg("Installed SIGUSR1 handler for traceback dump (kill -USR1 <pid>)")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

BASE_DIR = Path(__file__).resolve().parent  # .../app
PROJECT_ROOT = BASE_DIR.parent              # .../Sound to Sheet (maybe)

# ---- Boot debug prints (added) ----
_dbg(f"__file__={Path(__file__).resolve()}")
_dbg(f"BASE_DIR={BASE_DIR}")
_dbg(f"PROJECT_ROOT={PROJECT_ROOT}")
_dbg(f"cwd={Path.cwd()}")
_dbg(f"DEVICE={DEVICE}, cuda_available={torch.cuda.is_available()}")
try:
    if torch.cuda.is_available():
        _dbg(f"CUDA device_count={torch.cuda.device_count()}")
        _dbg(f"CUDA current_device={torch.cuda.current_device()}")
        _dbg(f"CUDA name={torch.cuda.get_device_name(torch.cuda.current_device())}")
except Exception as e:
    _dbg(f"CUDA info error: {e}")

_start_periodic_trace_dump()
_install_signal_dump()

CKPT_PATH = str(PROJECT_ROOT / "app" / "Transcription" / "2.result_hrplus" / "hrplus_best.pt")

MDX_DIR = PROJECT_ROOT / "app" / "Seperate" / "src" / "MDX" / "ckpt"
MDX_CKPT_PATH = str(MDX_DIR / "MDX23C_D1581.ckpt")
MDX_YAML_PATH = str(MDX_DIR / "model_2_stem_061321.yaml")

HTDE_CKPT = str(PROJECT_ROOT / "app" / "Seperate" / "src" / "htde" / "best.pth")

ACCOMP_CKPT = str(PROJECT_ROOT / "app" / "arrangement" / "ckpt" / "accomp_hybrid_best.pt")

# ---- Checkpoint path debug (added) ----
_print_path_info("CKPT_PATH (Transcription)", CKPT_PATH)
_print_path_info("MDX_CKPT_PATH", MDX_CKPT_PATH)
_print_path_info("MDX_YAML_PATH", MDX_YAML_PATH)
_print_path_info("HTDE_CKPT", HTDE_CKPT)
_print_path_info("ACCOMP_CKPT (Arrangement)", ACCOMP_CKPT)

# =========================
# accomp_service init (same logic, with debug wrappers added)
# =========================
_dbg("About to create accomp_service (this will load arrangement ckpt)")
_t0 = time.time()
try:
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
    _dbg(f"accomp_service created OK in {time.time() - _t0:.2f}s")
except Exception as e:
    _dbg(f"ERROR while creating accomp_service after {time.time() - _t0:.2f}s: {repr(e)}")
    traceback.print_exc()
    raise

# 추가: 업로드 검증/용량 제한 유틸

app = FastAPI()

# =========================
# Request logging middleware (added)
# =========================
@app.middleware("http")
async def log_requests(request, call_next):
    if not DEBUG_REQUESTS:
        return await call_next(request)

    rid = uuid.uuid4().hex[:8]
    t0 = time.time()
    try:
        print(f"[REQ {rid}] --> {request.method} {request.url.path}", flush=True)
        response = await call_next(request)
        dt = time.time() - t0
        print(f"[REQ {rid}] <-- {request.method} {request.url.path} {response.status_code} ({dt:.2f}s)", flush=True)
        return response
    except Exception as e:
        dt = time.time() - t0
        print(f"[REQ {rid}] !!  {request.method} {request.url.path} ERROR after {dt:.2f}s: {repr(e)}", flush=True)
        traceback.print_exc()
        raise


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_dbg("About to create transcriber (HRPlus)")
_t1 = time.time()
try:
    transcriber = HRPlusTranscriber(ckpt_path=CKPT_PATH, device=DEVICE)
    _dbg(f"transcriber created OK in {time.time() - _t1:.2f}s")
except Exception as e:
    _dbg(f"ERROR while creating transcriber after {time.time() - _t1:.2f}s: {repr(e)}")
    traceback.print_exc()
    raise

_dbg("About to create voice_extractor (MDX)")
_t2 = time.time()
try:
    voice_extractor = VoiceExtractor(
        VoiceExtractorConfig(
            ckpt_path=MDX_CKPT_PATH,
            yaml_path=MDX_YAML_PATH,
            device=str(DEVICE),
        )
    )
    _dbg(f"voice_extractor created OK in {time.time() - _t2:.2f}s")
except Exception as e:
    _dbg(f"ERROR while creating voice_extractor after {time.time() - _t2:.2f}s: {repr(e)}")
    traceback.print_exc()
    raise

_dbg("About to create piano_extractor (HTDE)")
_t3 = time.time()
try:
    piano_extractor = PianoExtractor(
        PianoExtractorConfig(
            ckpt_path=str(HTDE_CKPT),
            target_sr=22050,
            segment_sec=6.0,
            overlap_sec=1.0,
            device=str(DEVICE),
        )
    )
    _dbg(f"piano_extractor created OK in {time.time() - _t3:.2f}s")
except Exception as e:
    _dbg(f"ERROR while creating piano_extractor after {time.time() - _t3:.2f}s: {repr(e)}")
    traceback.print_exc()
    raise


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

        # ---- per-request debug (added) ----
        t0 = time.time()
        print(f"[TRANSCRIBE] start filename={file.filename} bytes={len(data)} path={in_path}", flush=True)

        midi_bytes = transcriber.transcribe_to_midi_bytes(in_path)

        dt = time.time() - t0
        print(f"[TRANSCRIBE] done in {dt:.2f}s -> midi_bytes={len(midi_bytes)}", flush=True)

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

        # ---- per-request debug (added) ----
        t0 = time.time()
        print(f"[MIDI2XML] start filename={file.filename} bytes={len(data)} mid_path={mid_path}", flush=True)

        # ✅ Grand Staff로 변환
        score = midi_to_grandstaff_score(str(mid_path), split_pitch=60)
        score.write("musicxml", fp=str(xml_path))

        xml_bytes = xml_path.read_bytes()

        dt = time.time() - t0
        print(f"[MIDI2XML] done in {dt:.2f}s -> xml_bytes={len(xml_bytes)}", flush=True)

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

        t0 = time.time()
        print(f"[SEP_INST] start filename={file.filename} bytes={len(data)} in={in_path}", flush=True)

        voice_extractor.extract_instrumental_to_wav(
            input_audio_path=str(in_path),
            output_wav_path=str(out_path),
            stem_name="Instrumental",
        )

        dt = time.time() - t0
        print(f"[SEP_INST] done in {dt:.2f}s -> out={out_path} bytes={out_path.stat().st_size if out_path.exists() else None}", flush=True)

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
        print(f"[SEP_INST] ERROR: {repr(e)}", flush=True)
        traceback.print_exc()
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

        t0 = time.time()
        print(f"[SEP_PIANO] start filename={file.filename} bytes={len(data)} in={in_path} has_vocals={has_vocals}", flush=True)

        # ✅ 핵심 분기
        piano_input = in_path
        if has_vocals:
            print("[MDX] extracting Instrumental…", flush=True)
            voice_extractor.extract_instrumental_to_wav(
                input_audio_path=str(in_path),
                output_wav_path=str(inst_path),
                stem_name="Instrumental",
            )
            piano_input = inst_path
            print(f"[SEP_PIANO] instrumental ready: {inst_path} exists={inst_path.exists()}", flush=True)

        print("[PIANO] extracting Piano…", flush=True)
        piano_extractor.extract_piano_to_wav(
            input_audio_path=str(piano_input),
            output_wav_path=str(out_path),
        )

        dt = time.time() - t0
        print(f"[SEP_PIANO] done in {dt:.2f}s -> out={out_path} bytes={out_path.stat().st_size if out_path.exists() else None}", flush=True)

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
    # validate_upload(file, MIDI_POLICY)

    # ✅ 용량 제한 걸고 읽기
    midi_bytes = await read_limited(file, MIDI_POLICY.max_bytes)

    t0 = time.time()
    print(f"[ARR] start filename={file.filename} midi_bytes={len(midi_bytes)} keep_original={keep_original}", flush=True)

    try:
        out_bytes = accomp_service.generate_midi_bytes(
            input_midi_bytes=midi_bytes,
            keep_original=keep_original,
        )
        dt = time.time() - t0
        print(f"[ARR] done in {dt:.2f}s -> out_midi_bytes={len(out_bytes)}", flush=True)

        return Response(
            content=out_bytes,
            media_type="audio/midi",
            headers={
                "Content-Disposition": 'inline; filename="accomp_pred.mid"',
                "Cache-Control": "no-store",
            },
        )
    except Exception as e:
        dt = time.time() - t0
        print(f"[ARR] ERROR after {dt:.2f}s: {repr(e)}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/ping")
def ping():
    # RunPod LB health check용: 최대한 가볍게 200만
    return Response(status_code=200)

@app.get("/health")
def health():
    return {"ok": True, "device": str(DEVICE)}