const API_BASE = "https://runpod-proxy-server.onrender.com";

/**
 * WAV/MP3/etc -> MIDI
 */
export async function transcribeToMidi(file) {
  const fd = new FormData();
  fd.append("file", file);

  const res = await fetch(`${API_BASE}/transcribe`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) throw new Error(await res.text());

  const buf = await res.arrayBuffer();
  return new Blob([buf], { type: "audio/midi" });
}

/**
 * MIDI -> MusicXML
 */
export async function midiToMusicXml(midiBlob) {
  const midiFile = new File([midiBlob], "transcribed.mid", { type: "audio/midi" });

  const fd = new FormData();
  fd.append("file", midiFile);

  const res = await fetch(`${API_BASE}/midi-to-musicxml`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) throw new Error(await res.text());

  const xmlBuf = await res.arrayBuffer();
  return new Blob([xmlBuf], { type: "application/vnd.recordare.musicxml+xml" });
}

/**
 * Separate Piano
 * - opts.hasVocals === true 이면:
 *   서버가 내부에서 MDX로 vocal 제거(instrumental 생성) 후 그걸로 피아노 분리
 * - 반환: piano.wav (Blob)
 */
export async function separatePiano(file, opts = {}) {
  const fd = new FormData();
  fd.append("file", file);

  // ✅ SeparateWorkspace에서 넘기는 { hasVocals }를 서버로 전달
  fd.append("has_vocals", String(!!opts.hasVocals));

  const res = await fetch(`${API_BASE}/separate/piano`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) throw new Error(await res.text());

  const buf = await res.arrayBuffer();
  return new Blob([buf], { type: "audio/wav" });
}

export async function arrangeMidi(midiBlobOrFile, opts = {}) {
  const keepOriginal = opts.keepOriginal ?? true;

  // 서버는 UploadFile로 받으니까 File 형태면 가장 깔끔함
  const midiFile =
    midiBlobOrFile instanceof File
      ? midiBlobOrFile
      : new File([midiBlobOrFile], "input.mid", { type: "audio/midi" });

  const fd = new FormData();
  fd.append("file", midiFile);
  fd.append("keep_original", String(!!keepOriginal)); // FastAPI Form(True/False)

  const res = await fetch(`${API_BASE}/arrangement/midi`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) throw new Error(await res.text());

  const buf = await res.arrayBuffer();
  return new Blob([buf], { type: "audio/midi" });
}
