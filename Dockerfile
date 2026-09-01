FROM node:22-bookworm-slim AS frontend
WORKDIR /src
COPY package.json pnpm-lock.yaml ./
RUN corepack enable && pnpm install --frozen-lockfile
COPY app ./app
COPY public ./public
COPY next.config.ts tsconfig.json postcss.config.mjs next-env.d.ts ./
RUN pnpm build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    METABALL_BIND_HOST=0.0.0.0 \
    METABALL_API_PORT=8780 \
    METABALL_PREVIEW_GRID=64
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY models ./models
COPY --from=frontend /src/out ./out
EXPOSE 8780
CMD ["python", "backend/inference_server.py"]
