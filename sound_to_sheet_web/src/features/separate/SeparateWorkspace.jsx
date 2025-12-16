import { useEffect, useRef, useState } from "react";
import FilePicker from "../../components/FilePicker";
import ErrorBox from "../../components/ErrorBox";
import { downloadBlob } from "../../components/utils";
import { separatePiano } from "../../api/sound2sheet";
import WaveSurfer from "wavesurfer.js";


export default function SeparateWorkspace() {
  const [file, setFile] = useState(null);
  const [hasVocals, setHasVocals] = useState(false);

  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState("");

  const [pianoBlob, setPianoBlob] = useState(null);
  const [pianoUrl, setPianoUrl] = useState("");

  // ✅ 오버레이/진행도
  const [showOverlay, setShowOverlay] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState("");

  const waveRef = useRef(null);
  const wsRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [curTime, setCurTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const ACCEPT_EXT = ["wav", "mp3", "flac"];
  const MAX_MB = 80;
  const MAX_SEC = 15 * 60;

  useEffect(() => {
    if (!pianoBlob) {
      setPianoUrl("");
      return;
    }
    const url = URL.createObjectURL(pianoBlob);
    setPianoUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [pianoBlob]);

  useEffect(() => {
    if (wsRef.current) {
      wsRef.current.destroy();
      wsRef.current = null;
    }
    setIsPlaying(false);
    setCurTime(0);
    setDuration(0);

    if (!pianoUrl || !waveRef.current) return;

    const ws = WaveSurfer.create({
      container: waveRef.current,
      height: 96,
      normalize: true,
      cursorWidth: 2,
      barWidth: 2,
      barGap: 2,
      barRadius: 2,
      waveColor: "rgba(255,255,255,.35)",
      progressColor: "rgba(255,255,255,.85)",
      cursorColor: "rgba(255,255,255,.9)",
    });

    wsRef.current = ws;
    ws.load(pianoUrl);

    ws.on("ready", () => {
      setDuration(ws.getDuration() || 0);
    });

    ws.on("audioprocess", () => {
      setCurTime(ws.getCurrentTime() || 0);
    });

    ws.on("seek", () => {
      setCurTime(ws.getCurrentTime() || 0);
    });

    ws.on("finish", () => {
      setIsPlaying(false);
    });

    return () => {
      ws.destroy();
      wsRef.current = null;
    };
  }, [pianoUrl]);

  function resetOutputs() {
    setPianoBlob(null);
  }

  function openOverlay(text) {
    setShowOverlay(true);
    setProgress(1);
    setProgressText(text);
  }

  function closeOverlaySoon() {
    setTimeout(() => {
      setShowOverlay(false);
      setProgress(0);
      setProgressText("");
    }, 350);
  }

  function getExt(name = "") {
    const m = name.toLowerCase().match(/\.([a-z0-9]+)$/);
    return m ? m[1] : "";
  }

  function getAudioDurationSec(file) {
    return new Promise((resolve) => {
      try {
        const url = URL.createObjectURL(file);
        const audio = new Audio();
        audio.preload = "metadata";
        audio.src = url;
        audio.onloadedmetadata = () => {
          URL.revokeObjectURL(url);
          const d = Number(audio.duration);
          resolve(Number.isFinite(d) ? d : null);
        };
        audio.onerror = () => {
          URL.revokeObjectURL(url);
          resolve(null);
        };
      } catch {
        resolve(null);
      }
    });
  }

  async function validateFileOrThrow(file) {
    if (!file) throw new Error("파일을 선택해주세요.");

    const ext = getExt(file.name);
    if (!ACCEPT_EXT.includes(ext)) {
      throw new Error(`지원하지 않는 형식입니다. (${ACCEPT_EXT.join(", ")})`);
    }

    const sizeMB = file.size / (1024 * 1024);
    if (sizeMB > MAX_MB) {
      throw new Error(`파일이 너무 큽니다. (${MAX_MB}MB 이하만 지원)`);
    }

    const dur = await getAudioDurationSec(file);
    if (dur && dur > MAX_SEC) {
      const min = Math.floor(MAX_SEC / 60);
      throw new Error(`오디오가 너무 깁니다. (${min}분 이하만 지원)`);
    }
  }

  function estimateSeconds(durationSec, fileSizeBytes) {
    const base = 2.0;
    const perSec = hasVocals ? 0.085 : 0.06;
    const sizeMB = fileSizeBytes / (1024 * 1024);
    const sizePenalty = Math.min(sizeMB * 0.03, 2.0);
    if (!durationSec) return 4.0 + sizePenalty;
    return base + durationSec * perSec + sizePenalty;
  }

  async function onRun() {
    if (!file) return;

    setError("");
    resetOutputs();

    try {
      openOverlay("파일을 확인하고 있습니다…");
      await validateFileOrThrow(file);
    } catch (e) {
      setError(e?.message || "파일 검증 실패");
      setShowOverlay(false);
      setProgress(0);
      setProgressText("");
      return;
    }

    setIsRunning(true);
    openOverlay(
      hasVocals
        ? "보컬 제거 후 피아노를 추출하고 있습니다…"
        : "피아노를 추출하고 있습니다…"
    );

    const dur = await getAudioDurationSec(file);
    const est = estimateSeconds(dur, file.size);
    const t0 = Date.now();

    const timer = setInterval(() => {
      const elapsed = (Date.now() - t0) / 1000;
      const r = Math.min(elapsed / Math.max(est, 1), 1);
      const target = Math.min(85, Math.floor(85 * r));
      setProgress((p) => Math.max(p, target));
    }, 450);

    const creep = setInterval(() => {
      setProgress((p) => (p < 95 ? p + 1 : p));
    }, 1700);

    try {
      setProgressText(
        hasVocals
          ? "보컬 제거 → 피아노 분리를 진행 중입니다…"
          : "피아노 분리를 진행 중입니다…"
      );

      const piano = await separatePiano(file, { hasVocals });
      setPianoBlob(piano);

      setProgress(100);
      setProgressText("완료!");
    } catch (e) {
      setError(e?.message || "분리 실패");
      setProgressText("에러가 발생했어요.");
    } finally {
      clearInterval(timer);
      clearInterval(creep);
      closeOverlaySoon();
      setIsRunning(false);
    }
  }

  function onDownloadPiano() {
    if (!pianoUrl) return;
    const base = file ? file.name.replace(/\.[^/.]+$/, "") : "audio";
    downloadBlob(pianoUrl, `${base}_piano.wav`);
  }

  function togglePlay() {
    const ws = wsRef.current;
    if (!ws) return;
    ws.playPause();
    setIsPlaying(ws.isPlaying());
  }

  function stopPlay() {
    const ws = wsRef.current;
    if (!ws) return;
    ws.stop();
    setIsPlaying(false);
    setCurTime(0);
  }

  function formatTime(sec) {
    const s = Math.max(0, Math.floor(sec || 0));
    const m = Math.floor(s / 60);
    const r = s % 60;
    return `${m}:${String(r).padStart(2, "0")}`;
  }

  return (
    <div className="app">
      {/* 로딩 오버레이 */}
      {showOverlay && (
        <div className="overlay no-print" role="status" aria-live="polite">
          <div className="overlayCard">
            <div className="overlayTitle">처리 중</div>
            <div className="overlayText">{progressText || "진행 중…"}</div>

            <div className="bar">
              <div className="barFill" style={{ width: `${progress}%` }} />
            </div>

            <div className="overlayMeta">
              <span className="spinner" aria-hidden />
              <span>{progress}%</span>
              <span className="dot" />
              <span>잠시만 기다려주세요</span>
            </div>
          </div>
        </div>
      )}

      <div className="shell">
        <header className="header no-print">
          <div className="brand">
            <div className="logo">🎵</div>
            <div>
              <div className="title">Sound to Sheet</div>
              <div className="subtitle">Audio → Separate → Piano</div>
            </div>
          </div>

          {/* ✅ 우측 끝 hover tip (Transcribe 스타일) */}
          <div className="headerTip">
            <span className="headerTipIcon">ⓘ</span>
            <div className="headerTipBubble" role="tooltip">
              <strong>전사 품질 안내</strong>
              <p><b>악기의 복잡함</b> 정도에 따라 품질이 변화합니다.<br />
              연주 음질, 잔향음에 따라 품질이 변화합니다.
              </p>
            </div>
          </div>
        </header>
        <section className="card no-print">
          <FilePicker
            onPick={(f) => {
              setFile(f);
              setError("");
              resetOutputs();
            }}
          />

          <div className="divider" />

          {/* ✅ 보컬 제거 토글 (직관형) */}
          <div className="rowBetween" style={{ marginBottom: 10 }}>
            <div>
              <div className="rowTitle">보컬 제거</div>
              <div className="rowDesc">
                켜면 <b>반주(피아노 중심)</b>으로 분리합니다
              </div>
            </div>

            <button
              type="button"
              className={`toggle ${hasVocals ? "on" : ""} ${isRunning ? "disabled" : ""}`}
              onClick={() => !isRunning && setHasVocals((v) => !v)}
              aria-pressed={hasVocals}
              aria-label="보컬 제거 토글"
            >
              <span className="toggleThumb" />
            </button>
          </div>

          {/* 토글 상태 텍스트 (한 줄로 명확하게) */}
          <div className="hintLine" style={{ marginBottom: 12 }}>
            현재: <b>{hasVocals ? "보컬 제거 ON (반주용)" : "원본 그대로 (보컬 포함 가능)"}</b>
          </div>

          <div className="tbBar" style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <button className="tbBtn tbBtnRed" disabled={!file || isRunning} onClick={onRun}>
              분리 실행
            </button>

            <button className="tbBtn" disabled={!pianoUrl} onClick={onDownloadPiano}>
              피아노 다운로드
            </button>
          </div>

          <ErrorBox error={error} />
          {/* ✅ 피아노 파형 미리듣기 */}
          {pianoUrl && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontWeight: 700, marginBottom: 6 }}>Piano Preview</div>

              {/* 파형 영역 */}
              <div
                ref={waveRef}
                style={{
                  width: "100%",
                  border: "1px solid rgba(255,255,255,.14)",
                  borderRadius: 14,
                  padding: 10,
                  background: "rgba(255,255,255,.04)",
                }}
              />

              {/* 컨트롤 */}
              <div
                style={{
                  display: "flex",
                  gap: 10,
                  alignItems: "center",
                  marginTop: 10,
                }}
              >
                <button className={`tbBtn tbBtnPlay`} onClick={togglePlay}>
                  {isPlaying ? "일시정지" : "재생"}
                </button>

                <button className="tbBtn tbBtnStop" onClick={stopPlay}>
                  정지
                </button>

                <div style={{ marginLeft: "auto", fontSize: 12, opacity: 0.7 }}>
                  {formatTime(curTime)} / {formatTime(duration)}
                </div>
                
              </div>
              
            </div>
            
          )}
        </section>
        {/* ⚠️ 데모 성능 안내 */}
        <section className="card">
        <div className="cardHead">
          <div className="cardTitle">주의사항</div>
        </div>
        <div className="cardHint">
          해당 기능은 <b>데모 버전</b>입니다.<br />
          연구 및 테스트 목적의 결과를 제공하며 개선중에 있습니다.
        </div>
      </section>
      </div>

      {/* ✅ Transcribe와 동일한 CSS 주입 */}
      <style>{css}</style>
    </div>
  );
}

const css = `
:root{
  --bg0:#070A12;
  --bg1:#0B1020;

  --card: rgba(255,255,255,.06);
  --card2: rgba(255,255,255,.08);
  --line: rgba(255,255,255,.10);

  --text: rgba(255,255,255,.92);
  --muted: rgba(255,255,255,.65);

  --shadow: 0 20px 60px rgba(0,0,0,.55);
  --r: 20px;

  /* ✅ Separate(갈색/노랑) 포인트 */
  --sep-a: rgba(180,110,60,.48);   /* brown ↓ */
  --sep-b: rgba(255,220,80,.42);   /* yellow ↓ */
  --sep-a2: rgba(180,110,60,.14);  /* brown soft ↓ */
  --sep-b2: rgba(255,220,80,.12);  /* yellow soft ↓ */
}

/* ✅ 배경 */
.app{
  min-height: 100vh;
  color: var(--text);
  padding-top: 24px;
  background:
    radial-gradient(1100px 680px at 18% 16%, var(--sep-a2), transparent 58%),
    radial-gradient(980px 620px at 86% 72%, var(--sep-b2), transparent 60%),
    radial-gradient(760px 420px at 55% 40%, rgba(255,255,255,0.05), transparent 66%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
}

.shell{
  max-width: 980px;
  margin: 36px auto;
  padding: 0 18px 60px;
}

/* ✅ Header */
.header{
  display:flex;
  align-items:center;
  justify-content:space-between;
  margin-bottom: 16px;

  position: relative;
  z-index: 50;
}

.brand{
  display:flex;
  gap: 12px;
  align-items:center;
}

.logo{
  width: 44px; height: 44px;
  display:grid; place-items:center;
  background: rgba(255,255,255,.08);
  border: 1px solid var(--line);
  border-radius: 14px;
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
}

.title{
  font-size: 20px;
  font-weight: 760;
  letter-spacing: -0.02em;
}

.subtitle{
  font-size: 12px;
  color: var(--muted);
  margin-top: 2px;
}

/* ✅ Card */
.card{
  background: var(--card);
  border: 1px solid rgba(255,220,80,.18);
  border-radius: var(--r);
  padding: 18px;
  box-shadow: var(--shadow), 0 0 40px rgba(255,220,80,.10);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  margin-top: 14px;

  /* ✅ 말풍선이 카드에 “가려지는” 문제 방지 */
  overflow: visible;
  position: relative;
  z-index: 1;
}

.divider{
  height: 1px;
  background: var(--line);
  margin: 14px 0;
}

/* ===== Loading overlay ===== */
.overlay{
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,.55);
  display: grid;
  place-items: center;
  z-index: 9999;
  padding: 18px;
}

.overlayCard{
  width: min(520px, 100%);
  background: rgba(20, 16, 12, .92);
  border: 1px solid rgba(255,220,80,.22);
  border-radius: 18px;
  box-shadow: 0 30px 90px rgba(0,0,0,.60), 0 0 50px rgba(255,220,80,.08);
  padding: 16px 16px 14px;
  backdrop-filter: blur(10px);
}

.overlayTitle{
  font-weight: 850;
  letter-spacing: -0.02em;
  font-size: 14px;
  opacity: .95;
}

.overlayText{
  margin-top: 6px;
  font-size: 13px;
  color: rgba(255,255,255,.70);
}

.bar{
  margin-top: 12px;
  height: 10px;
  border-radius: 999px;
  background: rgba(255,255,255,.08);
  overflow: hidden;
  border: 1px solid rgba(255,255,255,.10);
}

.barFill{
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(180,110,60,.85), rgba(255,220,80,.85));
  transition: width .25s ease;
}

.overlayMeta{
  margin-top: 10px;
  display: flex;
  gap: 8px;
  align-items: center;
  font-size: 12px;
  color: rgba(255,255,255,.60);
}

.overlayMeta .dot{
  width: 4px; height: 4px;
  border-radius: 999px;
  background: rgba(255,255,255,.35);
  display: inline-block;
}

.spinner{
  width: 12px;
  height: 12px;
  border-radius: 999px;
  border: 2px solid rgba(255,255,255,.25);
  border-top-color: rgba(255,220,80,.95);
  animation: spin .9s linear infinite;
}

@keyframes spin{
  to { transform: rotate(360deg); }
}

/* ===== Buttons ===== */
.tbBtn{
  appearance: none;
  border: 1px solid rgba(255,255,255,.14);
  background: rgba(255,255,255,.06);
  color: rgba(255,255,255,.88);
  padding: 10px 12px;
  border-radius: 14px;
  font-size: 13px;
  font-weight: 650;
  letter-spacing: -0.01em;
  cursor: pointer;
  transition: transform .08s ease, background .2s ease, border-color .2s ease, box-shadow .2s ease;
  box-shadow: 0 10px 26px rgba(0,0,0,.28);
}

.tbBtn:hover:not(:disabled){
  background: rgba(255,255,255,.09);
  border-color: rgba(255,255,255,.22);
  transform: translateY(-1px);
}
.tbBtn:active:not(:disabled){ transform: translateY(0px); }
.tbBtn:disabled{
  opacity: .45;
  cursor: not-allowed;
  box-shadow: none;
}

.tbBtnRed:not(:disabled){
  background: linear-gradient(180deg, rgba(180,110,60,.28), rgba(255,220,80,.12));
  border-color: rgba(255,220,80,.38);
  box-shadow: 0 14px 34px rgba(255,220,80,.10), 0 10px 26px rgba(0,0,0,.28);
}
.tbBtnRed:hover:not(:disabled){
  background: linear-gradient(180deg, rgba(180,110,60,.34), rgba(255,220,80,.16));
  border-color: rgba(255,220,80,.55);
}

/* ===== Toggle row ===== */
.rowBetween{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
}
.rowTitle{
  font-weight: 820;
  letter-spacing: -0.02em;
  font-size: 13px;
  color: rgba(255,255,255,.92);
}
.rowDesc{
  margin-top: 2px;
  font-size: 12px;
  color: rgba(255,255,255,.65);
}

/* ===== Toggle switch ===== */
.toggle{
  width: 54px;
  height: 32px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,.16);
  background: rgba(255,255,255,.06);
  box-shadow: 0 10px 26px rgba(0,0,0,.25);
  position: relative;
  padding: 0;
  cursor: pointer;
  transition: background .2s ease, border-color .2s ease, transform .08s ease;
}
.toggle:active{ transform: translateY(1px); }

.toggleThumb{
  width: 24px;
  height: 24px;
  border-radius: 999px;
  position:absolute;
  top: 50%;
  left: 4px;
  transform: translateY(-50%);
  background: rgba(255,255,255,.88);
  box-shadow: 0 8px 20px rgba(0,0,0,.35);
  transition: left .22s ease, background .2s ease;
}

.toggle.on{
  border-color: rgba(255,220,80,.55);
  background: linear-gradient(90deg, rgba(180,110,60,.35), rgba(255,220,80,.22));
}
.toggle.on .toggleThumb{
  left: 26px;
  background: rgba(255,255,255,.94);
}

.toggle.disabled{
  opacity: .45;
  cursor: not-allowed;
  box-shadow: none;
}

.hintLine{
  font-size: 12px;
  color: rgba(255,255,255,.60);
}
.hintLine b{ color: rgba(255,255,255,.86); }

/* ===== Player buttons ===== */
.tbBtnPlay:not(:disabled){
  background: linear-gradient(180deg, rgba(255,220,80,.22), rgba(255,220,80,.08));
  border-color: rgba(255,220,80,.35);
  box-shadow: 0 14px 34px rgba(255,220,80,.10), 0 10px 26px rgba(0,0,0,.28);
}
.tbBtnPlay:hover:not(:disabled){
  background: linear-gradient(180deg, rgba(255,220,80,.28), rgba(255,220,80,.10));
  border-color: rgba(255,220,80,.50);
}

.tbBtnStop:not(:disabled){
  background: linear-gradient(180deg, rgba(180,110,60,.18), rgba(180,110,60,.06));
  border-color: rgba(180,110,60,.35);
}
.tbBtnStop:hover:not(:disabled){
  background: linear-gradient(180deg, rgba(180,110,60,.24), rgba(180,110,60,.08));
  border-color: rgba(180,110,60,.50);
}

/* ===== Hover Tip (Transcribe 느낌, 아이콘 기준) ===== */
.headerTip{
  position: relative;
  z-index: 999; /* ✅ 헤더에서 최상단 */
}

.headerTipIcon{
  font-size: 14px;
  color: rgba(255,255,255,.55);
  cursor: help;
  padding: 6px;
  border-radius: 999px;
  transition: background .2s ease, color .2s ease;
}

.headerTipIcon:hover{
  background: rgba(255,255,255,.08);
  color: rgba(255,255,255,.9);
}

/* ✅ 아이콘 기준으로 “오른쪽 아래” 뜸 */
.headerTipBubble{
  position: absolute;
  top: 34px;       /* 아이콘 아래 */
  right: 0;        /* 아이콘 우측 정렬 */
  width: 280px;

  opacity: 0;
  transform: translateY(-6px);
  pointer-events: none;

  background: rgba(18,20,30,.96);
  border: 1px solid rgba(255,255,255,.14);
  border-radius: 14px;
  padding: 12px 14px;

  font-size: 12.5px;
  line-height: 1.5;
  color: rgba(255,255,255,.85);

  box-shadow: 0 20px 60px rgba(0,0,0,.45);
  transition: opacity .2s ease, transform .2s ease;

  z-index: 999999;
}

.headerTipBubble::before{
  content:"";
  position:absolute;
  top:-6px;
  right: 12px;
  width: 10px;
  height: 10px;
  transform: rotate(45deg);
  background: rgba(18,20,30,.96);
  border-left: 1px solid rgba(255,255,255,.10);
  border-top: 1px solid rgba(255,255,255,.10);
}

.headerTipBubble strong{
  display:block;
  margin-bottom: 6px;
  font-size: 12px;
  color: rgba(255,255,255,.92);
}

.headerTipBubble p{
  margin: 0;
  color: rgba(255,255,255,.78);
}

.headerTip:hover .headerTipBubble{
  opacity: 1;
  transform: translateY(0);
  pointer-events: auto;
}

/* print */
@media print{
  .no-print{ display:none !important; }
  .app{ background: #fff !important; color:#000 !important; padding-top:0 !important; }
  .card{ box-shadow:none !important; backdrop-filter:none !important; }
}

.cardHint {
  background-color: rgba(174, 127, 46, 0.1);
  color: rgba(254, 203, 120, 1);
  padding: 12px;
  border-radius: 8px;
  font-weight: 600;
  font-size: 14px;
  margin-top: 10px;
}

`;
