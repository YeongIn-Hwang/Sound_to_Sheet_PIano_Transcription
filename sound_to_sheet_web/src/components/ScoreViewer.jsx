import { useEffect, useMemo, useRef, useState } from "react";
import { OpenSheetMusicDisplay } from "opensheetmusicdisplay";

/**
 * ScoreViewer (개선판)
 * - OSMD(A4)로 만들어진 SVG "페이지들"을 각각 PNG로 변환 (큰 1장 캔버스 폭발 방지)
 * - 내부 저장은 dataURL 대신 Blob + objectURL (메모리/속도 개선)
 * - "모든 페이지 ZIP 다운로드" (가능하면) + 실패 시 다중 다운로드 fallback
 * - 이 컴포넌트는 서버에 아무것도 저장하지 않음 (브라우저 메모리에만 존재)
 */

function escapeXml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function sanitizeMusicXmlTitle(xmlText, title) {
  if (!xmlText) return xmlText;
  const safeTitle = escapeXml(title || "Untitled");

  xmlText = xmlText.replace(
    /<work-title>\s*(Music21 Fragment|music21\s*fragment)\s*<\/work-title>/gi,
    `<work-title>${safeTitle}</work-title>`
  );
  xmlText = xmlText.replace(
    /<movement-title>\s*(Music21 Fragment|music21\s*fragment)\s*<\/movement-title>/gi,
    `<movement-title>${safeTitle}</movement-title>`
  );

  // watermark 같은 문자열 정리(있으면)
  xmlText = xmlText.replace(/>Music21</g, ">Sound to Sheet<");
  xmlText = xmlText.replace(/>music21</g, ">Sound to Sheet<");

  return xmlText;
}

/** XML에 자잘한 깨짐이 있을 때 OSMD가 덜 민감하도록 정규화(필요한 만큼만) */
function normalizeMusicXML(xmlText) {
  if (!xmlText) return xmlText;
  let x = xmlText;

  // 종종 잘못된 measure width 같은 것들이 들어오는 케이스 방어
  x = x.replace(/(<measure\b[^>]*?)\swidth="[^"]*"/gi, "$1");

  return x;
}

function getSvgSize(svgEl) {
  if (!svgEl) return { w: 1, h: 1 };

  const vb = svgEl.getAttribute("viewBox");
  if (vb) {
    const parts = vb.split(/[\s,]+/).map(Number);
    if (parts.length === 4 && parts.every((n) => Number.isFinite(n))) {
      const w = Math.max(1, Math.ceil(parts[2]));
      const h = Math.max(1, Math.ceil(parts[3]));
      return { w, h };
    }
  }

  const wAttr = Number(svgEl.getAttribute("width"));
  const hAttr = Number(svgEl.getAttribute("height"));
  if (Number.isFinite(wAttr) && wAttr > 0 && Number.isFinite(hAttr) && hAttr > 0) {
    return { w: Math.ceil(wAttr), h: Math.ceil(hAttr) };
  }

  const rect = svgEl.getBoundingClientRect();
  return { w: Math.max(1, Math.ceil(rect.width)), h: Math.max(1, Math.ceil(rect.height)) };
}

/** SVG -> Canvas */
function svgToCanvas(svgEl, scale = 2) {
  return new Promise((resolve, reject) => {
    try {
      const { w, h } = getSvgSize(svgEl);

      const svgText = new XMLSerializer().serializeToString(svgEl);
      const svgBlob = new Blob([svgText], { type: "image/svg+xml;charset=utf-8" });
      const url = URL.createObjectURL(svgBlob);

      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = Math.ceil(w * scale);
        canvas.height = Math.ceil(h * scale);

        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.setTransform(scale, 0, 0, scale, 0, 0);
        ctx.drawImage(img, 0, 0, w, h);

        URL.revokeObjectURL(url);
        resolve(canvas);
      };
      img.onerror = (e) => {
        URL.revokeObjectURL(url);
        reject(e);
      };
      img.src = url;
    } catch (e) {
      reject(e);
    }
  });
}

function canvasToPngBlob(canvas) {
  return new Promise((resolve, reject) => {
    try {
      canvas.toBlob(
        (blob) => {
          if (!blob) reject(new Error("PNG Blob 생성 실패"));
          else resolve(blob);
        },
        "image/png",
        1.0
      );
    } catch (e) {
      reject(e);
    }
  });
}

/** 큰 Canvas를 A4 비율로 슬라이스해서 "페이지 캔버스들"로 반환 (fallback 용) */
function sliceCanvasByA4(canvas) {
  const A4_RATIO = 297 / 210; // height/width
  const pageW = canvas.width;
  const pageH = Math.floor(pageW * A4_RATIO);

  const pages = [];
  let y = 0;

  while (y < canvas.height) {
    const h = Math.min(pageH, canvas.height - y);

    const pageCanvas = document.createElement("canvas");
    pageCanvas.width = pageW;
    pageCanvas.height = h;

    const ctx = pageCanvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, pageW, h);
    ctx.drawImage(canvas, 0, y, pageW, h, 0, 0, pageW, h);

    pages.push(pageCanvas);
    y += h;
  }

  return pages;
}

/** 브라우저 다운로드 헬퍼 */
function downloadBlob(blobOrUrl, filename) {
  const url = typeof blobOrUrl === "string" ? blobOrUrl : URL.createObjectURL(blobOrUrl);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  if (typeof blobOrUrl !== "string") URL.revokeObjectURL(url);
}

async function downloadZip(blobs, filenames, zipName) {
  // jszip가 프로젝트에 있으면 사용, 없으면 fallback
  let JSZip;
  try {
    JSZip = (await import("jszip")).default;
  } catch {
    JSZip = null;
  }

  if (!JSZip) {
    // fallback: 여러 파일 다운로드 (브라우저가 막을 수 있음)
    for (let i = 0; i < blobs.length; i++) {
      downloadBlob(blobs[i], filenames[i]);
      // 너무 빠르게 누르면 브라우저가 막는 경우가 있어 아주 짧게 텀
      // eslint-disable-next-line no-await-in-loop
      await new Promise((r) => setTimeout(r, 120));
    }
    return { ok: false, reason: "no-jszip" };
  }

  const zip = new JSZip();
  for (let i = 0; i < blobs.length; i++) {
    zip.file(filenames[i], blobs[i]);
  }
  const zipBlob = await zip.generateAsync({ type: "blob" });
  downloadBlob(zipBlob, zipName);
  return { ok: true };
}

export default function ScoreViewer({ xmlBlob, title, onScoreReady, onPagesReady }) {
  const scoreDivRef = useRef(null);
  const osmdRef = useRef(null);

  // dataURL 대신 objectURL로 미리보기 (메모리/속도 개선)
  const [pageUrls, setPageUrls] = useState([]);
  const [pageBlobs, setPageBlobs] = useState([]);
  const [isRendering, setIsRendering] = useState(false);
  const [errMsg, setErrMsg] = useState("");

  // 품질(=스케일). 기본 2가 무난, 인쇄용이면 2~3
  const [quality, setQuality] = useState(2);

  const baseName = useMemo(() => {
    const n = (title || "score").trim() || "score";
    return n.replace(/[\\/:*?"<>|]+/g, "_");
  }, [title]);

  // objectURL 정리
  useEffect(() => {
    return () => {
      pageUrls.forEach((u) => {
        try {
          URL.revokeObjectURL(u);
        } catch {}
      });
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let alive = true;

    (async () => {
      try {
        if (!xmlBlob || !scoreDivRef.current) {
          // 초기화
          pageUrls.forEach((u) => URL.revokeObjectURL(u));
          setPageUrls([]);
          setPageBlobs([]);
          setErrMsg("");
          onScoreReady?.(false);
          return;
        }

        setIsRendering(true);
        // 이전 결과 정리
        pageUrls.forEach((u) => URL.revokeObjectURL(u));
        setPageUrls([]);
        setPageBlobs([]);
        setErrMsg("");
        onScoreReady?.(false);

        scoreDivRef.current.innerHTML = "";

        const osmd = new OpenSheetMusicDisplay(scoreDivRef.current, {
          autoResize: true,
          backend: "svg",
          pageFormat: "A4_P",
          drawingParameters: "default",
        });
        osmdRef.current = osmd;

        let xmlText = await xmlBlob.text();
        xmlText = sanitizeMusicXmlTitle(xmlText, title);
        xmlText = normalizeMusicXML(xmlText);

        await osmd.load(xmlText);

        // ⚠️ Zoom은 "SVG 페이지 수"에도 영향을 줄 수 있음.
        // 너무 크면 캔버스/메모리 터지니, UI/인쇄를 고려하면 0.40~0.60 정도가 안정적.
        osmd.Zoom = 0.45;

        await osmd.render();

        // 레이아웃 안정화: 2프레임
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

        const svgs = Array.from(scoreDivRef.current.querySelectorAll("svg"));
        if (!svgs.length) throw new Error("OSMD SVG를 찾지 못했습니다.");

        // ✅ 기본 전략: OSMD가 만든 "페이지 단위 svg"를 각각 PNG로 변환
        // (긴 1장 캔버스 만들지 않아서 안정적)
        let blobs = [];
        if (svgs.length > 1) {
          for (const s of svgs) {
            // eslint-disable-next-line no-await-in-loop
            const c = await svgToCanvas(s, quality);
            // eslint-disable-next-line no-await-in-loop
            const b = await canvasToPngBlob(c);
            blobs.push(b);
          }
        } else {
          // ⚠️ OSMD가 1장으로만 뽑는 케이스 fallback: A4 비율 슬라이스
          const bigCanvas = await svgToCanvas(svgs[0], quality);
          const pageCanvases = sliceCanvasByA4(bigCanvas);

          for (const pc of pageCanvases) {
            // eslint-disable-next-line no-await-in-loop
            const b = await canvasToPngBlob(pc);
            blobs.push(b);
          }
        }

        if (!alive) return;

        const urls = blobs.map((b) => URL.createObjectURL(b));
        setPageBlobs(blobs);
        setPageUrls(urls);
        onScoreReady?.(true);
        onPagesReady?.(blobs);

        // 오프스크린 DOM 정리(미리보기는 URL 기반이라 필요 없음)
        scoreDivRef.current.innerHTML = "";
      } catch (e) {
        console.error("ScoreViewer failed:", e);
        if (alive) {
          pageUrls.forEach((u) => URL.revokeObjectURL(u));
          setPageUrls([]);
          setPageBlobs([]);
          setErrMsg(e?.message || "악보 이미지를 만들지 못했습니다.");
          onScoreReady?.(false);
        }
      } finally {
        if (alive) setIsRendering(false);
      }
    })();

    return () => {
      alive = false;
      try {
        osmdRef.current?.clear?.();
        osmdRef.current?.dispose?.();
      } catch {}
      osmdRef.current = null;
      if (scoreDivRef.current) scoreDivRef.current.innerHTML = "";
    };
    // quality가 바뀌면 재생성(사용자가 품질 바꿔서 다시 뽑을 수 있게)
  }, [xmlBlob, title, quality]); // eslint-disable-line react-hooks/exhaustive-deps

  async function onDownloadAllZip() {
    if (!pageBlobs.length) return;
    const names = pageBlobs.map((_, i) => `${baseName}_p${String(i + 1).padStart(2, "0")}.png`);
    const zipName = `${baseName}_pages.zip`;

    const res = await downloadZip(pageBlobs, names, zipName);
    if (!res.ok && res.reason === "no-jszip") {
      // 사용자에게 "여러 개 다운로드" 경고를 한 번만 보여주고 싶다면 여기서 처리
      // (너무 수다스럽게 안 함)
      console.warn("[ScoreViewer] JSZip not found. Fallback to multi-download.");
    }
  }

  function onDownloadOne(idx) {
    if (!pageBlobs[idx]) return;
    downloadBlob(pageBlobs[idx], `${baseName}_p${String(idx + 1).padStart(2, "0")}.png`);
  }

  if (!xmlBlob) {
    return (
      <div
        style={{
          marginTop: 18,
          border: "1px dashed rgba(255,255,255,0.18)",
          borderRadius: 14,
          padding: 20,
          maxWidth: 780,
          marginLeft: "auto",
          marginRight: "auto",
          background: "rgba(255,255,255,0.04)",
          color: "rgba(255,255,255,0.75)",
        }}
      />
    );
  }

  return (
    <div
      style={{
        marginTop: 18,
        border: "1px solid rgba(0,0,0,0.12)",
        borderRadius: 14,
        padding: 18,
        maxWidth: 780,
        marginLeft: "auto",
        marginRight: "auto",
        background: "#fff",
        color: "#000",
      }}
    >
      {/* ✅ 오프스크린 렌더 (DOM에 있어야 SVG size가 안정적이라 완전 display:none은 피함) */}
      <div
        ref={scoreDivRef}
        style={{
          position: "absolute",
          left: "-10000px",
          top: 0,
          width: 780,
          opacity: 0,
          visibility: "visible",
          pointerEvents: "none",
        }}
      />

      {/* 상단 액션바 */}
      <div
        className="no-print"
        style={{
          display: "flex",
          gap: 10,
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 10,
        }}
      >
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <div style={{ fontSize: 12, color: "#666" }}>
            pages: <b>{pageUrls.length}</b>
          </div>

          <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 12, color: "#555" }}>
            품질
            <select
              value={quality}
              onChange={(e) => setQuality(Number(e.target.value))}
              style={{ padding: "6px 8px", borderRadius: 8, border: "1px solid rgba(0,0,0,0.18)" }}
              disabled={isRendering}
            >
              <option value={1.5}>빠름(1.5x)</option>
              <option value={2}>기본(2x)</option>
              <option value={3}>고해상도(3x)</option>
            </select>
          </label>
        </div>

        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
        </div>
      </div>

      {isRendering && (
        <div style={{ fontSize: 13, color: "#444" }} className="no-print">
          악보를 이미지로 만드는 중… (곡의 길이에 따라 더 걸릴 수 있습니다.)
        </div>
      )}

      {errMsg && !isRendering && (
        <div style={{ fontSize: 13, color: "#b00020" }} className="no-print">
          {errMsg}
        </div>
      )}

      {pageUrls.length > 0 && (
        <div>
          {pageUrls.map((src, idx) => (
            <div
              key={idx}
              style={{
                marginBottom: 14,
                border: "1px solid rgba(0,0,0,0.08)",
                borderRadius: 10,
                overflow: "hidden",
                background: "#fff",
              }}
            >
              <div
                className="no-print"
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  padding: "10px 10px",
                  borderBottom: "1px solid rgba(0,0,0,0.08)",
                  background: "rgba(0,0,0,0.02)",
                }}
              >
                <div style={{ fontSize: 12, color: "#444" }}>
                  {baseName} — page {idx + 1}
                </div>
                <button
                  onClick={() => onDownloadOne(idx)}
                  style={{
                    padding: "6px 10px",
                    borderRadius: 10,
                    border: "1px solid rgba(0,0,0,0.18)",
                    background: "#fff",
                    color: "#111",
                    cursor: "pointer",
                    fontSize: 12,
                  }}
                >
                  PNG 저장
                </button>
              </div>

              <img
                src={src}
                alt={`${baseName} page ${idx + 1}`}
                style={{ display: "block", width: "100%", height: "auto" }}
              />
            </div>
          ))}

          <div className="no-print" style={{ fontSize: 12, color: "#777", marginTop: 6 }}>
            * 저장된 악보 이미지는 서버에 업로드되지 않고, 이 페이지(탭)를 닫으면 사라질 수 있습니다.
          </div>
        </div>
      )}
    </div>
  );
}
