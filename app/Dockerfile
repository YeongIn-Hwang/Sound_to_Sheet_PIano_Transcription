# Python 3.10 버전의 이미지를 기반으로 사용
FROM python:3.10-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

COPY . .

EXPOSE 8000

# ✅ RunPod Load Balancer: PORT 환경변수로 리슨해야 함
CMD ["bash","-lc","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
