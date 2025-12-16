import { useEffect, useState } from "react";
import MidiFilePicker from "../../components/MidiFilePicker";  // 새로 만든 picker
import ErrorBox from "../../components/ErrorBox";
import { downloadBlob } from "../../components/utils";
import { arrangeMidi } from "../../api/sound2sheet";

export default function ArrangeWorkspace() {
  const [file, setFile] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState("");

  const [outBlob, setOutBlob] = useState(null);
  const [outUrl, setOutUrl] = useState("");

  // output Blob -> Object URL
  useEffect(() => {
    if (!outBlob) {
      setOutUrl("");
      return;
    }
    const url = URL.createObjectURL(outBlob);
    setOutUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [outBlob]);

  function resetOutputs() {
    setOutBlob(null);
  }

  async function onArrange() {
    if (!file) return;
    setError("");
    resetOutputs();

    setIsRunning(true);

    // ✅ 오버레이 먼저 렌더링되도록 “한 프레임 양보”
    await new Promise(requestAnimationFrame);
    // ✅ 추가로 이벤트루프 한 번 양보 (동기 작업 시작 전에 UI flush)
    await new Promise((r) => setTimeout(r, 0));

    try {
      const blob = await arrangeMidi(file, { keepOriginal: true });
      setOutBlob(blob);
    } catch (e) {
      setError(e?.message || "Arrangement failed");
    } finally {
      setIsRunning(false);
    }
  }

  function onDownload() {
    if (!outUrl) return;
    const base = file?.name?.replace(/\.[^/.]+$/, "") || "arranged";
    downloadBlob(outUrl, `${base}_accomp.mid`);
  }

  return (
    <div className="app">
      {/* 로딩 오버레이 */}
      {isRunning && (
        <div className="overlay">
          <div className="overlayCard">
            <div className="overlayTitle">처리 중</div>
            <div className="overlayText">반주를 생성 중입니다...</div>

            <div className="bar">
              <div className="barFill" style={{ width: "50%" }} />
            </div>

            <div className="overlayMeta">
              <span className="spinner" aria-hidden />
              <span>50%</span>
              <span className="dot" />
              <span>잠시만 기다려주세요</span>
            </div>
          </div>
        </div>
      )}

      <div className="shell">
        <header className="header">
          <div className="brand">
            <div className="logo">🎵</div>
            <div>
              <div className="title">Sound to Sheet</div>
              <div className="subtitle">MIDI → MIDI</div>
            </div>
          </div>
          {/* Tooltip Icon */}
          <div className="tipWrap">
            <span className="tipIcon">ⓘ</span>
            <div className="tipBubble">
              <strong>전사 품질 안내</strong>
              <p>
                코드가 정직하고 단순한 곡일수록 품질이 올라갑니다.
              </p>
            </div>
          </div>
        </header>

        <section className="card">
          <MidiFilePicker
            onPick={(f) => {
              setFile(f);
              setError("");
              resetOutputs();
            }}
          />
          <div className="divider" />
          <div className="toolbar">
            <button onClick={onArrange} disabled={isRunning || !file} className="tbBtn">
              음원변환
            </button>
            <button onClick={onDownload} disabled={!outUrl} className="tbBtn tbBtnPrimary">
              MIDI 다운로드
            </button>
          </div>
          <ErrorBox error={error} />
        </section>

        {/* 주의사항 추가 */}
        <section className="card">
          <div className="cardHead">
            <div className="cardTitle">주의사항</div>
          </div>
          <div className="cardHint">
            해당 기능은 데모입니다. 개선의 여지가 매우 많으며 악곡의 난이도에 따라 품질이 매우 달라질 수 있습니다.
          </div>
        </section>
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

  /* Red Accent (주황/빨강 포인트) */
  --tr-a: rgba(255,0,0,.60);   /* red */
  --tr-b: rgba(200,0,0,.46);   /* dark red */
  --tr-a2: rgba(255,0,0,.16);  /* red soft */
  --tr-b2: rgba(200,0,0,.12);  /* dark red soft */
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
  position: relative;  /* Tooltip을 위한 설정 */
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

/* 버튼 스타일 */
.tbBtn {
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

.tbBtn:active:not(:disabled){
  transform: translateY(0px);
}

.tbBtn:disabled{
  opacity: .45;
  cursor: not-allowed;
  box-shadow: none;
}

/* 강조된 Primary 버튼 스타일 */
.tbBtnPrimary {
  background: linear-gradient(180deg, rgba(255,0,0,.30), rgba(200,0,0,.10));
  border-color: rgba(200,0,0,.34);
  box-shadow: 0 14px 36px rgba(200,0,0,.10), 0 10px 26px rgba(0,0,0,.28);
}

.tbBtnPrimary:hover:not(:disabled){
  background: linear-gradient(180deg, rgba(255,0,0,.36), rgba(200,0,0,.14));
  border-color: rgba(200,0,0,.48);
}

/* Hover Tip */
.tipWrap{
  position: absolute;
  top: 0;
  right: 18px;
  z-index: 60;
}

.tipIcon{
  width: 22px;
  height: 22px;
  border-radius: 999px;
  display: grid;
  place-items: center;
  font-size: 13px;
  font-weight: 700;
  cursor: help;
  color: rgba(255,255,255,.70);
  background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.18);
  backdrop-filter: blur(6px);
}

.tipBubble{
  position: absolute;
  top: 140%;
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

/* 주의사항 스타일 */
.cardHint {
  background-color: rgba(255, 0, 0, 0.1);
  color: rgba(255, 103, 103, 1);
  padding: 12px;
  border-radius: 8px;
  font-weight: 600;
  font-size: 14px;
  margin-top: 10px;
}

.cardHint b {
  font-weight: 700;
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
  background: rgba(16, 10, 12, .92);
  border: 1px solid rgba(255,0,0,.22);
  border-radius: 18px;
  box-shadow: 0 30px 90px rgba(0,0,0,.60), 0 0 50px rgba(255,0,0,.08);
  padding: 16px 16px 14px;
  backdrop-filter: blur(10px);
}

.overlayTitle{ font-weight: 850; font-size: 14px; opacity: .95; }
.overlayText{ margin-top: 6px; font-size: 13px; color: rgba(255,255,255,.70); }

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
  background: linear-gradient(90deg, rgba(255,0,0,.85), rgba(200,0,0,.85));
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

.spinner{
  width: 12px;
  height: 12px;
  border-radius: 999px;
  border: 2px solid rgba(255,255,255,.25);
  border-top-color: rgba(255,0,0,.95);
  animation: spin .9s linear infinite;
}

@keyframes spin{ to { transform: rotate(360deg); } }
`;


