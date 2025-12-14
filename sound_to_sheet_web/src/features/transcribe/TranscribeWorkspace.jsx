import { useEffect, useState } from "react";
import FilePicker from "../../components/FilePicker";
import Toolbar from "../../components/Toolbar";
import ErrorBox from "../../components/ErrorBox";
import StatusText from "../../components/StatusText";
import ScoreViewer from "../../components/ScoreViewer";
import { downloadBlob } from "../../components/utils";
import { midiToMusicXml, transcribeToMidi } from "../../api/sound2sheet";

export default function Home() {
  const [file, setFile] = useState(null);
  const [isConverting, setIsConverting] = useState(false);
  const [error, setError] = useState("");

  const [midiBlob, setMidiBlob] = useState(null);
  const [midiUrl, setMidiUrl] = useState("");

  const [xmlBlob, setXmlBlob] = useState(null);
  const [xmlUrl, setXmlUrl] = useState("");

  const [isGeneratingScore, setIsGeneratingScore] = useState(false);
  const [isScoreReady, setIsScoreReady] = useState(false);

  // ✅ 오버레이/진행도
  const [showOverlay, setShowOverlay] = useState(false);
  const [progress, setProgress] = useState(0); // 0~100
  const [progressText, setProgressText] = useState(""); // 고정 문구

  const [scorePages, setScorePages] = useState([]);   // Blob[]
  const [scorePageCount, setScorePageCount] = useState(0);

  // ✅ 업로드 파일명 → 악보 제목
  const scoreTitle = file ? file.name.replace(/\.[^/.]+$/, "") : "";

  // Home.jsx 안 (컴포넌트 밖/안 아무데나 OK)
  const ACCEPT_EXT = ["wav", "mp3", "flac", "m4a", "ogg"];
  const MAX_MB = 40;          // 원하는 값으로
  const MAX_SEC = 12 * 60;    // 12분

  // MIDI Blob → Object URL
  useEffect(() => {
    if (!midiBlob) {
      setMidiUrl("");
      return;
    }
    const url = URL.createObjectURL(midiBlob);
    setMidiUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [midiBlob]);

  // XML Blob → Object URL
  useEffect(() => {
    if (!xmlBlob) {
      setXmlUrl("");
      return;
    }
    const url = URL.createObjectURL(xmlBlob);
    setXmlUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [xmlBlob]);

  // ✅ DEBUG: ScoreViewer로 넘길 값 확인
  useEffect(() => {
    console.log("[Home] xmlBlob:", xmlBlob);
    console.log("[Home] xmlBlob type:", xmlBlob?.type);
    console.log("[Home] xmlBlob size:", xmlBlob?.size);
    console.log("[Home] scoreTitle:", scoreTitle);
  }, [xmlBlob, scoreTitle]);

  function resetOutputs() {
    setMidiBlob(null);
    setXmlBlob(null);
    setIsScoreReady(false);
    setScorePages([]);
    setScorePageCount(0);
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

    async function validateFileOrThrow(file) {
    if (!file) throw new Error("파일을 선택해주세요.");

    // 1) 확장자
    const ext = getExt(file.name);
    if (!ACCEPT_EXT.includes(ext)) {
      throw new Error(`지원하지 않는 형식입니다. (${ACCEPT_EXT.join(", ")})`);
    }

    // 2) 용량
    const sizeMB = file.size / (1024 * 1024);
    if (sizeMB > MAX_MB) {
      throw new Error(`파일이 너무 큽니다. (${MAX_MB}MB 이하만 지원)`);
    }

    // 3) 길이(메타데이터)
    const dur = await getAudioDurationSec(file); // 네 코드에 이미 있음
    if (dur && dur > MAX_SEC) {
      const min = Math.floor(MAX_SEC / 60);
      throw new Error(`오디오가 너무 깁니다. (${min}분 이하만 지원)`);
    }

    // (옵션) 4) mime이 이상한 경우 가볍게 경고/차단
    // m4a/ogg는 브라우저마다 type이 비거나 다를 수 있어 "강제 차단"은 비추.
    // if (file.type && !file.type.startsWith("audio/")) { ... }
  }

  async function onConvert() {
    if (!file) return;
    setError("");
    resetOutputs();

    // ✅ 프론트 검증 먼저
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

    setIsConverting(true);

    // ✅ 오버레이 시작 (텍스트 고정)
    openOverlay("오디오를 분석하고 있습니다…");

    // duration 기반 예상시간
    const dur = await getAudioDurationSec(file);
    const est = estimateSeconds(dur, file.size);
    const t0 = Date.now();

    // ✅ 진행도만 업데이트 (텍스트는 안 바꿈)
    const timer = setInterval(() => {
      const elapsed = (Date.now() - t0) / 1000;
      const r = Math.min(elapsed / Math.max(est, 1), 1);
      const target = Math.min(85, Math.floor(85 * r));
      setProgress((p) => Math.max(p, target));
    }, 450);

    // 85~95%: 오래 걸릴 때도 “살아있게”
    const creep = setInterval(() => {
      setProgress((p) => (p < 95 ? p + 1 : p));
    }, 1700);

    try {
      const midi = await transcribeToMidi(file);
      setMidiBlob(midi);

      setProgress(100);
      setProgressText("완료!");
    } catch (e) {
      setError(e?.message || "변환 실패");
      setProgressText("에러가 발생했어요.");
    } finally {
      clearInterval(timer);
      clearInterval(creep);

      closeOverlaySoon();
      setIsConverting(false);
    }
  }

  function onDownloadMidi() {
    if (!midiUrl) return;
    downloadBlob(midiUrl, "transcribed.mid");
  }

  async function onDownloadScore() {
    if (!scorePages.length) return;

    // 파일명
    const baseName = (scoreTitle || "score").replace(/[\\/:*?"<>|]+/g, "_");
    const names = scorePages.map((_, i) => `${baseName}_p${String(i + 1).padStart(2, "0")}.png`);
    const zipName = `${baseName}_pages.zip`;

    // jszip 있으면 zip, 없으면 fallback 다중 다운로드
    let JSZip;
    try {
      JSZip = (await import("jszip")).default;
    } catch {
      JSZip = null;
    }

    if (!JSZip) {
      for (let i = 0; i < scorePages.length; i++) {
        const url = URL.createObjectURL(scorePages[i]);
        downloadBlob(url, names[i]);
        URL.revokeObjectURL(url);
        // eslint-disable-next-line no-await-in-loop
        await new Promise((r) => setTimeout(r, 120));
      }
      return;
    }

    const zip = new JSZip();
    scorePages.forEach((b, i) => zip.file(names[i], b));
    const zipBlob = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(zipBlob);
    downloadBlob(url, zipName);
    URL.revokeObjectURL(url);
  }

  async function onGenerateScore() {
  if (!midiBlob) return;
  setError("");

  setIsGeneratingScore(true);
  setIsScoreReady(false);
  setXmlBlob(null);

  openOverlay("악보를 생성하고 있습니다…");

  const est = 20.0;
  const t0 = Date.now();

  const timer = setInterval(() => {
    const elapsed = (Date.now() - t0) / 1000;
    const r = Math.min(elapsed / est, 1);
    const target = Math.min(90, Math.floor(90 * r));
    setProgress((p) => Math.max(p, target));
  }, 450);

  const creep = setInterval(() => {
    setProgress((p) => (p < 97 ? p + 1 : p));
  }, 1800);

  try {
    const xml = await midiToMusicXml(midiBlob);
    setXmlBlob(xml);

    setProgress(100);
    setProgressText("완료!");
  } catch (e) {
    setError(e?.message || "악보 생성 실패");
    setProgressText("에러가 발생했어요.");
  } finally {
    clearInterval(timer);
    clearInterval(creep);

    closeOverlaySoon();
    setIsGeneratingScore(false);
  }
}


  function onDownloadXml() {
    if (!xmlUrl) return;
    downloadBlob(xmlUrl, "score.musicxml");
  }

  function sliceCanvasByA4(canvas) {
    // A4 세로 비율 (297 / 210)
    const A4_RATIO = 297 / 210;

    const pageWidth = canvas.width;
    const pageHeight = Math.floor(pageWidth * A4_RATIO);

    const pages = [];
    let y = 0;

    while (y < canvas.height) {
      const h = Math.min(pageHeight, canvas.height - y);

      const pageCanvas = document.createElement("canvas");
      pageCanvas.width = pageWidth;
      pageCanvas.height = h;

      const ctx = pageCanvas.getContext("2d");
      ctx.drawImage(
        canvas,
        0, y, pageWidth, h,
        0, 0, pageWidth, h
      );

      pages.push(pageCanvas.toDataURL("image/png"));
      y += h;
    }

    return pages;
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

  function estimateSeconds(durationSec, fileSizeBytes) {
    const base = 1.0;        // 3.0 → 1.0 (오버헤드 줄이기)
    const perSec = 0.05;     // 0.12 → 0.05 (더 빠르게)
    const sizeMB = fileSizeBytes / (1024 * 1024);
    const sizePenalty = Math.min(sizeMB * 0.03, 1.5); // 0.06/3 → 0.03/1.5

    if (!durationSec) return 2.5 + sizePenalty;       // 8 → 2.5
    return base + durationSec * perSec + sizePenalty;
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
              <div className="subtitle">Audio → MIDI → MusicXML → Sheet</div>
            </div>
          </div>

          {/* ✅ 우측 끝 hover tip */}
          <div className="headerTip">
            <span className="headerTipIcon">ⓘ</span>

            <div className="headerTipBubble">
              <strong>전사 품질 안내</strong>
              <p>
                순수 <b>피아노 곡일수록</b> 전사 성능이 올라갑니다.<br />
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

          <Toolbar
            file={file}
            isConverting={isConverting}
            midiBlob={midiBlob}
            isGeneratingScore={isGeneratingScore}
            xmlBlob={xmlBlob}
            isScoreReady={isScoreReady}
            onDownloadScore={onDownloadScore}
            onConvert={onConvert}
            onDownloadMidi={onDownloadMidi}
            onGenerateScore={onGenerateScore}
            onDownloadXml={onDownloadXml}
            scorePageCount={scorePageCount}
          />

          <ErrorBox error={error} />
          {midiBlob && (
          <div className="qualityNote no-print">
            ※ 자동 전사 특성상 일부 박자/음표 오차가 있을 수 있습니다. 최종 편집은 MuseScore 사용을 권장합니다.
            <br></br>탭을 닫으면 작업이 모두 사라집니다. 
          </div>
        )}
        </section>

        <section className="card">
          <div className="cardHead no-print">
            <div className="cardTitle">악보 미리보기</div>
            <div className="cardHint">
              변환 후 <b>악보 생성</b>을 누르면 여기에 표시됩니다.
            </div>
          </div>
        </section>
        <ScoreViewer
          xmlBlob={xmlBlob}
          title={scoreTitle}
          onScoreReady={setIsScoreReady}
          onPagesReady={(blobs) => {          // ✅ 추가
            setScorePages(blobs || []);
            setScorePageCount(blobs?.length || 0);
          }}
        />
      </div>

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

  /* ✅ Transcribe(파랑/민트) 포인트 */
  --tr-a: rgba(70,140,255,.60);   /* blue */
  --tr-b: rgba(60,255,210,.46);   /* mint */
  --tr-a2: rgba(70,140,255,.16);  /* blue soft */
  --tr-b2: rgba(60,255,210,.12);  /* mint soft */
}

.app{
  min-height: 100vh;
  color: var(--text);
  padding-top: 24px;
  background:
    radial-gradient(1200px 700px at 20% 10%, var(--tr-a2), transparent 55%),
    radial-gradient(900px 600px at 80% 30%, var(--tr-b2), transparent 55%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
}

.shell{
  max-width: 980px;
  margin: 36px auto;
  padding: 0 18px 60px;
}

.header{
  display:flex;
  align-items:center;
  justify-content:space-between;
  margin-bottom: 16px;
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

.card{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: var(--r);
  padding: 18px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  margin-top: 14px;
}

.divider{
  height: 1px;
  background: var(--line);
  margin: 14px 0;
}

.cardHead{
  display:flex;
  align-items:baseline;
  justify-content:space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.cardTitle{
  font-size: 15px;
  font-weight: 700;
}

.cardHint{
  font-size: 12px;
  color: var(--muted);
}

@media (max-width: 640px){
  .cardHead{ flex-direction: column; align-items:flex-start; }
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
  background: rgba(16, 18, 28, .92);
  border: 1px solid rgba(255,255,255,.14);
  border-radius: 18px;
  box-shadow: 0 30px 90px rgba(0,0,0,.60);
  padding: 16px 16px 14px;
  backdrop-filter: blur(10px);
}

.overlayTitle{
  font-weight: 800;
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
  /* ✅ 진행바: 파랑 → 민트 */
  background: linear-gradient(90deg, rgba(70,140,255,.88), rgba(60,255,210,.82));
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

/* ===== mini spinner ===== */
.spinner{
  width: 12px;
  height: 12px;
  border-radius: 999px;
  border: 2px solid rgba(255,255,255,.25);
  border-top-color: rgba(60,255,210,.90); /* ✅ 민트 */
  animation: spin .9s linear infinite;
}

@keyframes spin{
  to { transform: rotate(360deg); }
}

.qualityNote{
  margin-top: 12px;
  font-size: 12.5px;
  color: rgba(255, 255, 255, 0.85);
  line-height: 1.55;
  padding: 12px 14px;
  border-radius: 14px;
  /* ✅ 안내 박스: 파랑/민트 톤 */
  background: linear-gradient(
    180deg,
    rgba(70,140,255,.12),
    rgba(60,255,210,.04)
  );
  border: 1px solid rgba(60,255,210,.22);
  border-left: 4px solid rgba(60,255,210,.70);
  box-shadow: 0 12px 32px rgba(0,0,0,.35);
}

/* ===== Toolbar buttons ===== */
.tbBar .tbBtn{
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

.tbBar .tbBtn:hover:not(:disabled){
  background: rgba(255,255,255,.09);
  border-color: rgba(255,255,255,.22);
  transform: translateY(-1px);
}

.tbBar .tbBtn:active:not(:disabled){
  transform: translateY(0px);
}

.tbBar .tbBtn:disabled{
  opacity: .45;
  cursor: not-allowed;
  box-shadow: none;
}

/* ✅ Primary(악보 ZIP 다운로드): 파랑/민트 포인트 */
.tbBar .tbBtnPrimary:not(:disabled){
  background: linear-gradient(180deg, rgba(70,140,255,.30), rgba(60,255,210,.10));
  border-color: rgba(60,255,210,.34);
  box-shadow: 0 14px 36px rgba(60,255,210,.10), 0 10px 26px rgba(0,0,0,.28);
}
.tbBar .tbBtnPrimary:hover:not(:disabled){
  background: linear-gradient(180deg, rgba(70,140,255,.36), rgba(60,255,210,.14));
  border-color: rgba(60,255,210,.48);
}

/* ✅ Red(악보 생성): 너무 빨갛게 튀지 않게 “코랄”로만 살짝 */
.tbBar .tbBtnRed:not(:disabled){
  background: linear-gradient(180deg, rgba(255,120,90,.22), rgba(255,120,90,.08));
  border-color: rgba(255,120,90,.34);
  box-shadow: 0 14px 34px rgba(255,120,90,.10), 0 10px 26px rgba(0,0,0,.28);
}
.tbBar .tbBtnRed:hover:not(:disabled){
  background: linear-gradient(180deg, rgba(255,120,90,.28), rgba(255,120,90,.10));
  border-color: rgba(255,120,90,.46);
}
  /* ===== Hover Tip (Top-right) ===== */
.tipWrap{
  position: fixed;
  top: 14px;
  right: 18px;
  z-index: 20;
}

.tipIcon{
  width: 22px;
  height: 22px;
  border-radius: 999px;
  display: grid;
  place-items: center;
  font-size: 13px;
  font-weight: 700;
  cursor: default;

  color: rgba(255,255,255,.70);
  background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.18);
  backdrop-filter: blur(6px);
}

.tipBubble{
  position: absolute;
  top: 30px;
  right: 0;

  min-width: 240px;
  padding: 10px 12px;
  border-radius: 12px;

  background: rgba(16,18,28,.92);
  border: 1px solid rgba(255,255,255,.14);
  box-shadow: 0 18px 50px rgba(0,0,0,.45);

  font-size: 12.5px;
  line-height: 1.55;
  color: rgba(255,255,255,.78);

  opacity: 0;
  transform: translateY(-4px);
  pointer-events: none;
  transition: opacity .18s ease, transform .18s ease;
}

.tipBubble strong{
  display: block;
  margin-bottom: 4px;
  font-size: 12px;
  color: rgba(255,255,255,.92);
}

.tipWrap:hover .tipBubble{
  opacity: 1;
  transform: translateY(0);
}

.header{
  display: flex;
  align-items: center;
  justify-content: space-between; /* ← 핵심 */
}

/* 우측 끝 */
.headerTip{
  position: relative;
}

.headerTipIcon{
  font-size: 14px;
  color: rgba(255,255,255,.55);
  cursor: help;
  padding: 6px;
  border-radius: 50%;
  transition: background .2s ease, color .2s ease;
}

.headerTipIcon:hover{
  background: rgba(255,255,255,.08);
  color: rgba(255,255,255,.9);
}

/* 말풍선 */
.headerTipBubble{
  position: absolute;
  top: 140%;
  right: 0;
  width: 270px;

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
}

.headerTip:hover .headerTipBubble{
  opacity: 1;
  transform: translateY(0);
}

/* 헤더를 기준으로 z-index가 제대로 먹게 */
.header{
  position: relative;
  z-index: 50;
}

/* 툴팁 래퍼도 위로 */
.tipWrap{
  position: relative;
  z-index: 60;
}

/* 툴팁 자체를 최상단으로 */
.tipBubble{
  z-index: 9999;
}

`;

