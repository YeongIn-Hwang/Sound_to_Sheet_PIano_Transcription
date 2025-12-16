# 🎵 Sound to Sheet

**Sound to Sheet** is a web-based AI service that converts user-uploaded audio into  
piano transcriptions, separated piano audio, and MIDI accompaniment.  
No additional installation is required — everything runs directly in your web browser.

---

## 🌐 Web Page Link
👉 https://sound-to-sheet-p-iano-transcription-three.vercel.app

---

## ⏰ Service Availability
- **Operating Hours**: **09:00 AM – 10:00 PM (KST)**
- Requests may be limited outside operating hours to protect server resources.

---

## 🚀 How to Use

1. Prepare the **audio or MIDI file** you want to convert.
   - Supported formats: `wav`, `mp3`, `flac`, `m4a`, `ogg`, `mid`

2. Visit the web page and upload your file.

3. Select the desired feature.
   - Piano Transcription (Audio → Sheet / MIDI)
   - Piano Separation (Audio → Piano)
   - MIDI Arrangement (MIDI → MIDI Accompaniment)

4. Once processing is complete, download the results.
   - MIDI files or separated piano audio
   - Generated MIDI files can be opened directly in notation software such as MuseScore.

---

## ⚠️ Notes

- This service is provided as a **demo version**.
- Output quality may vary depending on musical complexity, chord structure, and performance style.
- Simpler and clearer chord progressions generally produce better transcription and arrangement results.
- Generated outputs are recommended for **research and educational purposes**.

---

## 🛠 Tech Stack

- **Frontend**: React + Vite  
- **Backend**: FastAPI  
- **AI Models**
  - Piano Transcription (Onsets & Frames / HRPlus-based)
  - Audio Source Separation (U-Net / Demucs-based)
  - MIDI Arrangement (Transformer-based)

---

## 📌 Disclaimer

This project is a personal research and educational project.  
Commercial use is not guaranteed.
