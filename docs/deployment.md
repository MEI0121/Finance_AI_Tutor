# Deployment (Docker)

This project is deployed with two containers:
- `backend` (FastAPI + LangGraph + RAG ingestion)
- `frontend` (Next.js production server)

The stack is orchestrated via `docker-compose.yml` in the repository root.

## Prerequisites

- Docker Desktop (or Docker Engine + Compose plugin)
- An OpenAI API key

## Environment

Create a root `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
```

`docker compose` automatically injects this into the backend service.

## Build and Run

From the repository root:

```bash
docker compose up --build
```

Access:
- Frontend: `http://localhost:3000`
- Backend health: `http://localhost:8000/health`

## Important Runtime Behavior

### Backend image (`Dockerfile.backend`)

- Uses `python:3.11-slim`
- Installs dependencies from root `requirements.txt`
- Copies `backend/` and `data/` into the image
- **Runs ingestion before API startup**:
  - `python -m backend.ingest_pdf`
  - then `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`

This guarantees Chroma is (re)built before serving requests.

### Frontend image (`Dockerfile.frontend`)

- Multi-stage build (`node:20-alpine`)
- Runs `npm ci`, `npm run build`, then `npm run start`
- Exposes port `3000`

## Persistent Vector Store

Chroma persistence is mapped with a host volume:

```yaml
volumes:
  - ./backend/chroma_db:/app/backend/chroma_db
```

This keeps embeddings/index data between container restarts.

## Stop / Reset

Stop services:

```bash
docker compose down
```

Stop and remove containers + network + anonymous volumes:

```bash
docker compose down -v
```

Note: The bind mount `./backend/chroma_db` is a host directory and remains unless you delete it manually.
