# ── web ──────────────────────────────────────────────────────────────────────
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm ci || npm install
COPY web/ ./
RUN npm run build

# ── service ──────────────────────────────────────────────────────────────────
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 AI_DATA_DIR=/data
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY tulpar_ai ./tulpar_ai
COPY skills ./skills
COPY corpus ./corpus
COPY fixtures ./fixtures
COPY evals ./evals
COPY tools ./tools
COPY --from=web /web/dist ./web/dist
RUN python -c "from tulpar_ai.api.app import app; assert app.routes"
EXPOSE 8089
CMD ["python", "-m", "tulpar_ai"]
