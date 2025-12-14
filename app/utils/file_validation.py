# src/app/utils/file_validation.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Set

from fastapi import UploadFile, HTTPException

@dataclass(frozen=True)
class UploadPolicy:
    max_bytes: int
    allowed_ext: Set[str]
    allowed_content_types: Optional[Set[str]] = None


# ✅ 기본 정책 (너 상황에 맞게 숫자만 바꿔 쓰면 됨)
TRANSCRIBE_POLICY = UploadPolicy(
    max_bytes=25 * 1024 * 1024,  # 25MB
    allowed_ext={".wav", ".mp3", ".flac", ".m4a", ".ogg"},
    allowed_content_types={
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/flac",
        "audio/x-flac",
        "audio/mp4",
        "audio/x-m4a",
        "audio/aac",
        "audio/ogg",
        "application/ogg",
    },
)

MIDI_POLICY = UploadPolicy(
    max_bytes=10 * 1024 * 1024,  # 10MB면 충분 (필요하면 ↑)
    allowed_ext={".mid", ".midi"},
    allowed_content_types={
        "audio/midi",
        "audio/x-midi",
        "application/x-midi",
        "application/octet-stream",  # 브라우저/환경에 따라 이렇게 오기도 함
    },
)


def _ext_of(filename: str) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def validate_upload(file: UploadFile, policy: UploadPolicy) -> None:
    """
    1) 확장자 검증
    2) content-type(있는 경우) 검증
    """
    ext = _ext_of(file.filename or "")
    if ext not in policy.allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext or '(no extension)'}",
        )

    # content_type은 브라우저/프록시에서 비거나 애매하게 올 때가 있어서 "있으면"만 체크
    ct = (file.content_type or "").lower().strip()
    if policy.allowed_content_types and ct:
        if ct not in policy.allowed_content_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type: {ct}",
            )


async def read_limited(file: UploadFile, max_bytes: int) -> bytes:
    """
    업로드 파일을 안전하게 읽되, max_bytes를 넘으면 413으로 차단.
    - UploadFile.read() 한 번에 다 읽는 것보다 안전 (대용량/악성 업로드 방어)
    """
    total = 0
    chunks: list[bytes] = []

    # ✅ 이미 다른 데서 read() 했으면 포인터가 끝에 있을 수 있음
    #    (FastAPI UploadFile은 seek 지원)
    try:
        await file.seek(0)
    except Exception:
        pass

    while True:
        chunk = await file.read(1024 * 1024)  # 1MB
        if not chunk:
            break

        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max allowed: {max_bytes // (1024 * 1024)}MB",
            )
        chunks.append(chunk)

    if total == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    return b"".join(chunks)
