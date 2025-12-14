# Node.js 최신 버전으로 업그레이드
FROM node:20

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사
COPY package.json package-lock.json ./

# 종속성 설치
RUN npm install

# 코드 복사
COPY . .

# 빌드 실행
RUN npm run build

# 포트 설정
EXPOSE 3000

# 앱 실행
CMD ["npm", "start"]
