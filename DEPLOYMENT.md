# Public deployment

The application is a single-origin service: Python serves the exported Next.js
frontend, `/infer`, `/eigenvalue`, videos, and paper assets on port 8780.

## Docker

```bash
cp .env.example .env
docker compose up -d --build
```

The service listens on port 8780. Permit that port in the server firewall and
map a domain or public IP to the server if remote users should access it.

## Native Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\start_public.ps1
```

## Public-service controls

- `METABALL_RATE_LIMIT`: requests allowed per client IP and window.
- `METABALL_RATE_WINDOW`: rate-limit window in seconds.
- `METABALL_MAX_CONCURRENT`: simultaneous inference/eigenvalue computations.
- `METABALL_MAX_BODY_BYTES`: maximum POST request size.
- `METABALL_ALLOWED_ORIGINS`: comma-separated browser origins or `*`.

The defaults allow 30 requests per IP per 60 seconds and two simultaneous
computations. Excess requests receive HTTP 429; excess concurrency receives 503.

## Large artifacts

The three neural epoch-2000 checkpoints are required by the online service.
The 6,144-to-2,048 Ritz archive is larger than GitHub's normal per-file limit
and is intentionally excluded because it is not used by `/infer`.
