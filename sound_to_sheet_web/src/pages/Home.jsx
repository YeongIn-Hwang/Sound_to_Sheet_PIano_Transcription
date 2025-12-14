import { useEffect, useMemo, useRef, useState } from "react";
import "./home.css";
import TranscribeWorkspace from "../features/transcribe/TranscribeWorkspace";
import SeparateWorkspace from "../features/separate/SeparateWorkspace";
import ArrangeWorkspace from "../features/arrange/ArrangeWorkspace";

function scrollToId(id) {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
}

function AudioPlayer({ audioSrc, label }) {
  return (
    <div className="audio-player">
      <p>{label}</p>
      <audio controls>
        <source src={audioSrc} type="audio/mpeg" />
        <source src={audioSrc} type="audio/mp4" />
        Your browser does not support the audio element.
      </audio>
    </div>
  );
}

function FeatureSection({ id, title, desc, bullets, cta, badge, audioFiles }) {
  const isThreeAudio = audioFiles.length === 3; // 3개의 음원일 경우 스타일 다르게

  return (
    <section id={id} className="snapSection">
      <div className="sectionInner">
        <div className="left">
          <div className="badge">{badge}</div>
          <h1 className="h1">{title}</h1>
          <p className="p">{desc}</p>

          <ul className="ul">
            {bullets.map((b, i) => (
              <li key={i} className="li">
                {b}
              </li>
            ))}
          </ul>

          <div className="ctaRow">
            {cta?.disabled ? (
              <button className="btn disabled" disabled>
                Coming soon
              </button>
            ) : (
              <button className="btn" onClick={cta?.onClick}>
                {cta?.label || "Try it"}
              </button>
            )}

            <button className="btn ghost" onClick={() => scrollToId("top")}>
              Back to top
            </button>
          </div>
        </div>

        <div className="right">
          {/* 음원이 3개일 경우 스타일 다르게 */}
          {isThreeAudio ? (
            <div className="audio-player-container-3">
              {audioFiles.map((audio, index) => (
                <AudioPlayer key={index} audioSrc={`/assets/audio/${audio.file}`} label={audio.label} />
              ))}
            </div>
          ) : (
            <div className="audio-player-container">
              {audioFiles.map((audio, index) => (
                <AudioPlayer key={index} audioSrc={`/assets/audio/${audio.file}`} label={audio.label} />
              ))}
            </div>
          )}
          <p className="musescore-info">실제 해당 서비스를 사용하여 MuseScore를 통해 녹음한 결과입니다.</p> {/* 추가된 문구 */}
        </div>
      </div>
    </section>
  );
}


export default function Home() {
  const containerRef = useRef(null);
  const [active, setActive] = useState("top");
  const [openTranscribe, setOpenTranscribe] = useState(false);
  const [openSeparate, setOpenSeparate] = useState(false);
  const [openArrange, setOpenArrange] = useState(false);

  const sections = useMemo(
    () => [
      { id: "top", label: "Home" },
      { id: "transcribe", label: "Transcribe" },
      { id: "separate", label: "Separate" },
      { id: "arrange", label: "Arrange" },
    ],
    []
  );

  useEffect(() => {
    const root = containerRef.current;
    if (!root) return;

    const ids = sections.map((s) => s.id);
    const els = ids.map((id) => document.getElementById(id)).filter(Boolean);
    if (!els.length) return;

    const io = new IntersectionObserver(
      (entries) => {
        // 가장 많이 보이는 섹션을 active로
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible?.target?.id) setActive(visible.target.id);
      },
      { root, threshold: [0.25, 0.5, 0.75] }
    );

    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, [sections]);

  return (
    <>
      {/* Quickbar */}
      <div className="quickbar">
        <div className="brand" onClick={() => scrollToId("top")} role="button" tabIndex={0}>
          🎵 Sound to Sheet
        </div>

        <div className="nav">
          {sections.map((s) => (
            <button
              key={s.id}
              className={`navBtn ${active === s.id ? "active" : ""}`}
              onClick={() => scrollToId(s.id)}
            >
              {s.label}
            </button>
          ))}
        </div>

        <div className="rightSlot">
          <button className="navBtn pill" onClick={() => scrollToId("transcribe")}>
            Start
          </button>
        </div>
      </div>

      {/* Scroll-snap container */}
      <div ref={containerRef} className="snapContainer">
        <div className="bgStage" aria-hidden="true">
          <div className={`bgLayer hero ${active === "top" ? "on" : ""}`} />
          <div className={`bgLayer trans ${active === "transcribe" ? "on" : ""}`} />
          <div className={`bgLayer sep ${active === "separate" ? "on" : ""}`} />
          <div className={`bgLayer arr ${active === "arrange" ? "on" : ""}`} />
        </div>
        {/* Hero */}
        <section id="top" className="snapSection hero">
          <div className="heroInner">
            <div className="heroText">
              <div className="badge">Piano Audio → MIDI → Sheet</div>
              <h1 className="heroTitle">Turn piano recordings into sheet music.</h1>
              <p className="heroSub">
                피아노 연주 음원을 업로드하면 
                <br />
                MIDI와 MusicXML을 생성하고 악보로 바로 미리볼 수 있습니다.
              </p>

              <div className="ctaRow">
                <button className="btn" onClick={() => scrollToId("transcribe")}>
                  Go to Transcribe
                </button>
                <button className="btn ghost" onClick={() => scrollToId("separate")}>
                  See Separation
                </button>
              </div>

            </div>

            <div className="heroCard">
              <div className="cardTitle">What you get</div>
              <div className="cardGrid">
                <div className="miniCard">
                  <div className="miniTitle">MIDI</div>
                  <div className="miniDesc">다운로드 / 재생</div>
                </div>
                <div className="miniCard">
                  <div className="miniTitle">MusicXML</div>
                  <div className="miniDesc">악보 렌더 / 편집</div>
                </div>
                <div className="miniCard">
                  <div className="miniTitle">PNG</div>
                  <div className="miniDesc"> 악보로 저장</div>
                </div>
                <div className="miniCard">
                  <div className="miniTitle">Stems</div>
                  <div className="miniDesc">분리</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 3 sections */}
        <FeatureSection
          id="transcribe"
          badge="01"
          title="Transcribe"
          desc="피아노 연주 음원을 업로드하면  피아노 MIDI를 생성하고, MusicXML로 변환해 악보를 얻습니다."
          bullets={[
            "Upload piano audio → Piano MIDI",
            "Piano MIDI → MusicXML → Sheet preview",
            "MIDI / MusicXML / PDF export",
          ]}
          audioFiles={[
            { file: "Transcribe.m4a", label: "Original 1" },
            { file: "Transcribe_.m4a", label: "Transcribe 1" },
            { file: "Transcribe2.m4a", label: "Original 2" },
            { file: "Transcribe2_.m4a", label: "Transcribe 2" },
          ]}
          cta={{
            label: "Try Transcribe",
            onClick: () => setOpenTranscribe(true),
          }}
        />

        <FeatureSection
          id="separate"
          badge="02"
          title="Separate"
          desc="곡에서 피아노 소스를 분리하여 보다 효과적인 전사를 위한 stem을 제공합니다."
          bullets={[
            "Piano Stem separation",
            "Cleaner transcription input",
            "Download separated audio",
          ]}
          audioFiles={[
            { file: "Separate.m4a", label: "Original 1" },
            { file: "Separate_.m4a", label: "Separate 1" },
            { file: "Separate2.m4a", label: "Original 2" },
            { file: "Separate2_.m4a", label: "Separate 2" },
          ]}
          cta={{
            label: "Try Separate",
            onClick: () => setOpenSeparate(true),
          }}
        />

        <FeatureSection
          id="arrange"
          badge="03"
          title="Arrange (DEMO)"
          desc="전사된 MIDI의 반주를 더해주는 편곡 도구를 제공합니다."
          bullets={[
            "MIDI Data Processing",
            "Accompaniment Instrument Assignment",
            "MIDI Format Optimization",
          ]}
          audioFiles={[
            { file: "Arrangement.m4a", label: "Original Sound" },
            { file: "Arrangement_.m4a", label: "Arranged Accompaniment" },
            { file: "Arrangement__.m4a", label: "Piano accompaniment with melody" },
          ]}
          cta={{
            label: "Try Arrange",
            onClick: () => setOpenArrange(true),
          }}
        />
      </div>

      {openTranscribe && (
        <div className="modalOverlay" role="dialog" aria-modal="true">
          <div className="modalPanel">
            <div className="modalTop">
              <div className="modalTitle">Transcribe</div>
              <button className="navBtn" onClick={() => setOpenTranscribe(false)}>
                Close
              </button>
            </div>
            <div className="modalBody">
              <TranscribeWorkspace />
            </div>
          </div>
        </div>
      )}

      {openSeparate && (
        <div className="modalOverlay" role="dialog" aria-modal="true">
          <div className="modalPanel">
            <div className="modalTop">
              <div className="modalTitle">Separate</div>
              <button className="navBtn" onClick={() => setOpenSeparate(false)}>
                Close
              </button>
            </div>
            <div className="modalBody">
              <SeparateWorkspace />
            </div>
          </div>
        </div>
      )}

      {openArrange && (
        <div className="modalOverlay" role="dialog" aria-modal="true">
          <div className="modalPanel">
            <div className="modalTop">
              <div className="modalTitle">Arrange</div>
              <button className="navBtn" onClick={() => setOpenArrange(false)}>
                Close
              </button>
            </div>
            <div className="modalBody">
              <ArrangeWorkspace />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
