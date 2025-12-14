export default function Toolbar({
  file,
  isConverting,
  midiBlob,
  isGeneratingScore,
  xmlBlob,
  isScoreReady,
  onConvert,
  onDownloadMidi,
  onGenerateScore,
  onDownloadXml,
  onDownloadScore,
  scorePageCount,
}) {
  const canDownloadScore = isScoreReady && scorePageCount > 0;

  return (
    <div className="no-print tbBar" style={{ marginTop: 16, display: "flex", gap: 12, flexWrap: "wrap" }}>
      <button className="tbBtn" disabled={!file || isConverting} onClick={onConvert}>
        {isConverting ? "변환 중..." : "변환하기"}
      </button>

      <button className="tbBtn" disabled={!midiBlob} onClick={onDownloadMidi}>
        MIDI 다운로드
      </button>

      {/* ✅ 악보 생성(=변환) 버튼: 붉은 포인트 */}
      <button className="tbBtn tbBtnRed" disabled={!midiBlob || isGeneratingScore} onClick={onGenerateScore}>
        {isGeneratingScore ? "악보 생성 중..." : "악보 생성"}
      </button>

      <button className="tbBtn" disabled={!xmlBlob} onClick={onDownloadXml}>
        XML 다운로드
      </button>

      {/* ✅ 최종 산출물: ZIP 다운로드는 살짝 보라/블루 포인트 */}
      <button
        className="tbBtn tbBtnPrimary"
        disabled={!canDownloadScore}
        onClick={onDownloadScore}
      >
        악보 다운로드{scorePageCount ? ` (${scorePageCount}p)` : ""} · ZIP
      </button>
    </div>
  );
}
