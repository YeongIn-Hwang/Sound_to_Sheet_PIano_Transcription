# Python 3.10 버전의 이미지를 기반으로 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# PYTHONPATH 환경 변수 설정
ENV PYTHONPATH=/app

# 의존성 파일 복사
COPY requirements.txt .

# PyTorch와 torchaudio 설치를 위한 추가 인덱스 URL 사용 (한 줄로 합침)
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# 나머지 코드 복사
COPY . .

# 포트 설정 (FastAPI가 8000 포트를 사용한다고 가정)
EXPOSE 8000

# FastAPI 애플리케이션 실행 (app.app:app으로 변경)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
