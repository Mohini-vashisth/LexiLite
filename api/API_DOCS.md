# LexiLite REST API Documentation

## Overview

Django REST API for legal document analysis with:
- ✅ **Real-time clause analysis** with risk classification (Sentence-BERT + logistic regression)
- ✅ **Confidence gating** — low-confidence calls routed to review, not guessed
- ✅ **PDF/DOCX upload** alongside raw-text analysis
- ✅ **Authentication** — every endpoint except `/health/` and `/metrics/` requires a real user (LEX-6); each user only ever sees their own analysis history
- ✅ **Health checks** and metrics monitoring
- ⚠️ Latency/concurrency: see [Performance Characteristics](#performance-characteristics) for real measured numbers and their caveats — the "sub-200ms, 4 workers" figures once claimed here were never actually measured

## Quick Start

### 1. Setup

```bash
cd api
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser   # or any User — this is what you'll authenticate as
```

### 2. Development Server

```bash
python manage.py runserver
```

Then either open **http://localhost:8000/** and log in via the "Log in" link (session auth, for the browser page), or grab your API token for programmatic access:

```bash
python manage.py shell -c "from rest_framework.authtoken.models import Token; print(Token.objects.get(user__username='<you>').key)"
```

(A token is created automatically for every user the moment the account exists — see `analyzer/signals.py` — no separate "generate token" step.)

### 3. Production Server (Gunicorn)

```bash
gunicorn --workers=4 --bind=0.0.0.0:8000 config.wsgi:application
```

### 4. Benchmark

```bash
export LEXILITE_API_TOKEN=<your token from step 2>
python benchmark.py --concurrent 50 --total 200
```

---

## Authentication

Two ways in, chosen automatically by however the client identifies itself:

- **Session auth** — for the browser page at `/`. Log in at `/api-auth/login/?next=/`, log out at `/api-auth/logout/`. POST requests from the page must include an `X-CSRFToken` header read from the `csrftoken` cookie (standard Django AJAX pattern) — the page already does this.
- **Token auth** — for scripts/API clients. Send `Authorization: Token <key>` on every request. No login flow, no cookies, no CSRF concerns.

`/api/health/` and `/api/metrics/` are the only endpoints that stay open to anyone — they're operational probes (a load balancer has no user to authenticate as), and they reveal nothing beyond aggregate up/down and latency numbers.

Every `AnalysisResult` is tied to whoever created it (`owner`, required). `/api/documents/recent/` and the standard list/retrieve actions only ever return the authenticated caller's own records — there's no way to see another user's analysis history through this API, regardless of which auth method you used.

## API Endpoints

All endpoints below require authentication (see [Authentication](#authentication)) except `/api/health/` and `/api/metrics/`. An unauthenticated request to any of the others gets `403 {"detail": "Authentication credentials were not provided."}`.

### 1. Analyze Document

**POST** `/api/documents/analyze/`

Analyze a legal document for risky clauses.

**Request:**
```json
{
  "text": "Full document text here...",
  "filename": "contract.pdf"
}
```

**Response (201 Created):**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "contract.pdf",
  "total_clauses": 42,
  "risky_count": 3,
  "risky_clauses": [
    {
      "clause_id": 5,
      "text": "This clause limits our liability in unforeseen...",
      "risk": "risky",
      "confidence": 0.92,
      "safe_prob": 0.08,
      "risky_prob": 0.92
    }
  ],
  "processing_time_ms": 145.32,
  "api_response_time_ms": 152.18
}
```

**Error Response (400):**
```json
{
  "error": "Document text required (min 10 characters)"
}
```

---

### 2. Upload Document (PDF/DOCX)

**POST** `/api/documents/upload/`

Analyze an uploaded PDF or DOCX file — same response shape as `/analyze`, but the text is extracted from the file server-side instead of being sent as JSON.

**Request:** `multipart/form-data` with a `file` field.

```bash
curl -X POST http://localhost:8000/api/documents/upload/ \
  -F "file=@contract.pdf"
```

**Response (201 Created):** identical shape to `/analyze` (see above) — `filename` is taken from the uploaded file's name.

**Error Responses (400):**
```json
{ "error": "No file provided — send it as multipart/form-data under the 'file' field" }
```
```json
{ "error": "Unsupported file type 'text/plain' for 'notes.txt'. Only .pdf and .docx are supported." }
```
```json
{ "error": "File too large (15.2MB) — max 10MB" }
```
```json
{ "error": "Could not read PDF: ..." }
```

**Limits:** 10MB max file size (`MAX_UPLOAD_SIZE_BYTES` env var to override). Processing is synchronous — a very large document ties up the request for its full processing time; there's no async/queued path yet (tracked separately).

---

### 3. Get Recent Analyses

**GET** `/api/documents/recent/?limit=10`

Retrieve recent document analyses.

**Response:**
```json
{
  "count": 10,
  "results": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "contract.pdf",
      "risky_count": 3,
      "created_at": "2024-08-27T10:30:45Z"
    }
  ]
}
```

---

### 4. Health Check

**GET** `/api/health/`

Check API and model health.

**Response (200 or 503):**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "timestamp": 1724758245.123456
}
```

---

### 5. Performance Metrics

**GET** `/api/metrics/`

Get performance metrics.

**Response:**
```json
{
  "total_requests": 1523,
  "avg_latency_ms": 147.32,
  "max_latency_ms": 385.21,
  "p50_latency_ms": "See monitoring dashboard",
  "message": "Average latency: 147.32ms (1523 requests)"
}
```

---

## Performance Characteristics

**Measured 2026-09-22** (`benchmark.py --concurrent 50 --total 100`, 30-clause document, Windows + RTX 5060 Ti, GPU-accelerated `torch+cu128`). See the main [README's Performance section](README.md#performance) for the full writeup and caveats — summarized here:

```
Success rate:                100/100 (100%)
API-only latency (mean):     331.4ms
API-only latency (median):   102.8ms
API-only latency (p95):      1366.5ms
API-only latency (p99):      2009.2ms
Throughput:                  24.5 req/sec
```

**This was measured against `manage.py runserver` (Django's dev server), not Gunicorn.** The "4 worker processes" throughput/latency figures that used to be here were never measured and have been removed rather than left as a guess. Gunicorn doesn't run on Windows at all (no native process forking), so a real multi-worker number needs Linux (WSL or Docker) — not done yet, not blocking, tracked as a follow-up rather than blocking this doc on it.

No resource-usage (memory/CPU) numbers are published here — they were never actually measured, only guessed at previously.

---

## Deployment Options

### Docker

```bash
docker build -t lexilite-api .
docker run -p 8000:8000 lexilite-api
```

### Production Checklist

- [ ] Set `DEBUG=False` in settings
- [ ] Configure `ALLOWED_HOSTS`
- [ ] Use a production database (PostgreSQL recommended)
- [ ] Set up Redis for caching
- [ ] Enable HTTPS/SSL
- [ ] Configure CORS properly
- [ ] Set up monitoring/logging (Sentry, Datadog, etc.)
- [ ] Use a reverse proxy (Nginx/HAProxy)
- [ ] Set up auto-scaling based on load

### Nginx Reverse Proxy Config

```nginx
upstream gunicorn {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name api.lexilite.com;
    client_max_body_size 10M;

    location / {
        proxy_pass http://gunicorn;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    location /api/health/ {
        proxy_pass http://gunicorn;
        access_log off;
    }
}
```

---

## Client Examples

### Python

```python
import requests

response = requests.post(
    'http://localhost:8000/api/documents/analyze/',
    json={
        'text': 'Your document text...',
        'filename': 'contract.pdf'
    }
)

data = response.json()
print(f"Risky clauses found: {data['risky_count']}")
```

### cURL

```bash
curl -X POST http://localhost:8000/api/documents/analyze/ \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your document text...",
    "filename": "contract.pdf"
  }'
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8000/api/documents/analyze/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'Your document text...',
    filename: 'contract.pdf'
  })
});

const data = await response.json();
console.log(`Risky clauses: ${data.risky_count}`);
```

---

## Troubleshooting

### Models not loading

```bash
python manage.py shell
from analyzer.inference import ModelCache
cache = ModelCache()  # Forces reload
```

### High latency

1. Check CPU usage - may need more workers
2. Monitor memory - may need more RAM
3. Check disk I/O - SSD recommended
4. Verify network - may have network bottleneck

### Out of memory

```bash
# Reduce worker count
gunicorn --workers=2 ...

# Or increase swap/RAM
```

---

## Monitoring & Logging

### Application Logs

```bash
# View logs
tail -f /var/log/lexilite-api.log

# Log levels: INFO, WARNING, ERROR, CRITICAL
```

### Performance Monitoring

```python
# Get current metrics
GET /api/metrics/

# Expected output shows:
# - Total requests processed
# - Average latency
# - Max latency observed
# - Throughput
```

### Health Monitoring

```bash
# Check every 30 seconds
watch -n 30 'curl -s http://localhost:8000/api/health/ | jq'
```

---

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP/REST
       ▼
┌─────────────────────────┐
│   Nginx (Reverse Proxy) │
└──────┬──────────────────┘
       │
       ├─────────────────────┐
       │                     │
    ┌──▼──────────┐   ┌──────▼──────┐
    │ Gunicorn    │   │ Gunicorn    │
    │ Worker 1    │   │ Worker 2    │
    │ (sync)      │   │ (sync)      │
    └──┬──────────┘   └──┬───────────┘
       │                 │
       └────────┬────────┘
                ▼
        ┌──────────────────┐
        │  Django App      │
        ├──────────────────┤
        │ ModelCache       │ (Singleton)
        │ - SBERT loaded   │
        │ - Classifier     │
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │ LegalAnalyzer    │
        │ - Extract        │
        │ - Encode         │
        │ - Predict        │
        │ - Track Metrics  │
        └──────────────────┘
```

---

## Performance Report (actual, not a template)

The block below was previously a fabricated example presented as an achieved result — replaced with what was actually measured. Re-run `benchmark.py` and update this whenever the setup changes (different hardware, Gunicorn instead of the dev server, etc.) rather than reverting to invented numbers.

```
LexiLite REST API — Performance Report
Measured: 2026-09-22

Test Configuration
  Concurrent requests: 50
  Total requests: 100 (capped by DRF's 100/hour anon throttle — see below)
  Server: manage.py runserver (Django dev server, NOT Gunicorn)
  Hardware: Windows, NVIDIA RTX 5060 Ti, torch+cu128 (GPU-accelerated)

Results
  Success rate:              100/100 (100%)
  API latency (mean):        331.4ms
  API latency (median):      102.8ms
  API latency (p95):         1366.5ms
  API latency (p99):         2009.2ms
  Throughput:                24.5 req/sec

NOT achieved at these settings: sub-200ms mean latency, or a true
100+ concurrent production benchmark (the dev server serializes GPU
work rather than truly parallelizing it — see README.md#performance
for why). A representative production number needs Gunicorn's
multi-worker setup, which requires Linux (WSL/Docker) since Gunicorn
doesn't run on Windows.
```

---

## License

MIT
