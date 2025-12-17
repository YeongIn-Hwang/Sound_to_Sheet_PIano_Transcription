import { useId, useRef, useState } from "react";

const ACCEPT = ".wav,.mp3,.flac";

export default function FilePicker({ onPick }) {
  const inputId = useId();
  const inputRef = useRef(null);
  const [isOver, setIsOver] = useState(false);
  const [pickedName, setPickedName] = useState("");

  function pickFile(f) {
    if (!f) return;
    setPickedName(f.name);
    onPick?.(f);
  }

  function onChange(e) {
    const f = e.target.files?.[0];
    pickFile(f);
  }

  function onDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    setIsOver(false);

    const f = e.dataTransfer?.files?.[0];
    if (f) pickFile(f);
  }

  function onDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    setIsOver(true);
  }

  function onDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    setIsOver(false);
  }

  return (
    <div className="no-print" style={{ marginTop: 14 }}>
      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        style={{
          width: "100%",
          boxSizing: "border-box",            // ✅ 추가
          overflow: "hidden",                 // ✅ 추가 (삐져나옴 차단)
          borderRadius: 16,
          border: `1px dashed ${
            isOver ? "rgba(180,150,90,0.55)" : "rgba(255,255,255,0.18)"
          }`,
          background: isOver
            ? "rgba(180,150,90,0.10)"
            : "rgba(255,255,255,0.04)",
          padding: "22px 18px",
          cursor: "pointer",
          transition: "200ms ease",
          outline: "none",
        }}
      >
        {/* ✅ wrap은 유지해도 되지만, 핵심은 minWidth:0 + ellipsis */}
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <div
            style={{
              width: 42,
              height: 42,
              borderRadius: 12,
              display: "grid",
              placeItems: "center",
              background: "rgba(255,255,255,0.06)",
              border: "1px solid rgba(255,255,255,0.12)",
              fontSize: 18,
              flex: "0 0 auto",
            }}
          >
            🎹
          </div>

          {/* ✅ minWidth: 240 때문에 폭이 못 줄어서 버튼이 밖으로 밀려남 → minWidth:0 */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div
              style={{
                fontWeight: 900,
                opacity: 0.95,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {pickedName ? `선택됨: ${pickedName}` : "피아노 오디오를 여기에 드롭하세요"}
            </div>

            <div
              style={{
                marginTop: 4,
                opacity: 0.7,
                fontSize: 13,
                lineHeight: 1.4,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              Drag & Drop 또는 클릭해서 파일 선택 · 지원 형식: {ACCEPT}
            </div>
          </div>

          {/* ✅ 버튼은 줄어들지 않게 고정 + 줄바꿈 금지 */}
          <div
            style={{
              flex: "0 0 auto",               // ✅ 추가
              whiteSpace: "nowrap",           // ✅ 추가
              height: 36,
              padding: "0 14px",
              borderRadius: 12,
              border: "1px solid rgba(255,255,255,0.14)",
              background: "rgba(255,255,255,0.08)",
              display: "grid",
              placeItems: "center",
              fontWeight: 800,
              opacity: 0.95,
              userSelect: "none",
            }}
          >
            파일 선택
          </div>
        </div>
      </div>

      <input
        id={inputId}
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        style={{ display: "none" }}
        onChange={onChange}
      />
    </div>
  );
}
