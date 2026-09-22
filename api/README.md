# LexiLite REST API

Production-ready Django REST API for real-time legal document analysis.

## Features

✅ **Real-time analysis** - Clause risk classification via Sentence-BERT + logistic regression
✅ **Batch processing** - Efficient model encoding
✅ **Confidence gating** - Low-confidence calls routed to review instead of guessed (see [Confidence gating](#confidence-gating) below)
✅ **Performance monitoring** - Built-in metrics & health checks
✅ **PDF/DOCX upload** - Not just raw text
✅ **Containerized** - Docker build verified end-to-end (see [Performance](#performance) for what's measured vs. what's still aspirational)

## Quick Start

### Development

```bash
# Setup
python manage.py migrate

# Run
python manage.py runserver

# Test
python benchmark.py
```

### Production

```bash
# Using Gunicorn (4 workers)
gunicorn --workers=4 --bind=0.0.0.0:8000 config.wsgi:application

# Or using Docker
docker build -t lexilite-api .
docker run -p 8000:8000 lexilite-api
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/documents/analyze/` | Analyze document for risky clauses |
| GET | `/api/documents/recent/` | Get recent analyses |
| GET | `/api/health/` | Health check |
| GET | `/api/metrics/` | Performance metrics |

## Example Usage

```bash
curl -X POST http://localhost:8000/api/documents/analyze/ \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your contract text...",
    "filename": "contract.pdf"
  }'
```

## Performance

**Measured 2026-09-22** — real numbers from `benchmark.py --concurrent 50 --total 100`, run against a live server on Windows with an NVIDIA RTX 5060 Ti (GPU-accelerated inference, `torch+cu128`). The `--total 200` originally planned here isn't achievable against a single anonymous client without changes — DRF's default throttle caps anonymous requests at 100/hour, and the throttle counts every attempt (see `settings.py`'s `AnonRateThrottle`).

**Important caveat:** this was measured against Django's development server (`manage.py runserver`), **not** the Gunicorn/4-worker setup described elsewhere in this doc. The dev server is largely single-process, so concurrent requests queue rather than truly running in parallel — the numbers below reflect that, not the production architecture. Gunicorn doesn't run natively on Windows (it depends on Unix process forking), so a true multi-worker benchmark needs Linux (WSL or Docker) and is tracked separately, not blocking here.

```
Requests:              100/100 succeeded (100%)
API-only latency (mean):    331.4ms
API-only latency (median):  102.8ms
API-only latency (p95):     1366.5ms
API-only latency (p99):     2009.2ms
Throughput:                 24.5 req/sec
```

The wide spread between median and p95/p99 is real, not noise: the first ~30-50 concurrent requests show steadily *climbing* latency (GPU work queueing behind a single CUDA stream, likely compounded by CUDA kernel warm-up), then later requests in the same batch drop to 13-30ms once the GPU is warm. A production deployment with proper multi-process serving would behave differently — this specific number describes the dev server under load, not a ceiling on the architecture.

## Documentation

See [API_DOCS.md](API_DOCS.md) for:
- Complete API documentation
- Deployment guides
- Performance benchmarking
- Architecture diagrams
- Troubleshooting guide

## Architecture

```
Client → Nginx → Gunicorn Workers (4) → Django → ModelCache → Inference
```

**Key optimizations:**
- Singleton model cache (loaded once at startup)
- Batch encoding (32 items per batch)
- Thread pool for concurrent requests
- In-memory caching layer
- Efficient memory management

## Monitoring

```bash
# Health status
curl http://localhost:8000/api/health/ | jq

# Performance metrics
curl http://localhost:8000/api/metrics/ | jq

# Load testing (50 concurrent, 200 total requests)
python benchmark.py --concurrent 50 --total 200
```

## Configuration

### Environment Variables

```bash
DEBUG=False                          # Disable debug mode
DJANGO_SECRET_KEY=your-secret-key   # Production secret
ALLOWED_HOSTS=api.lexilite.com      # Allowed domains
```

### Gunicorn Tuning

```bash
# For 100+ concurrent requests
gunicorn \
  --workers=4                    # CPU cores
  --worker-class=sync           # Use sync workers
  --bind=0.0.0.0:8000          # Bind address
  --timeout=30                  # 30s timeout
  --max-requests=1000          # Reload worker after N requests
  --max-requests-jitter=100    # Randomize reload
  config.wsgi:application
```

## Deployment Checklist

- [ ] Set `DEBUG=False`
- [ ] Configure `ALLOWED_HOSTS`
- [ ] Use PostgreSQL for production
- [ ] Set up Redis caching
- [ ] Enable HTTPS/SSL
- [ ] Configure CORS origins
- [ ] Set up Nginx reverse proxy
- [ ] Enable application monitoring (Sentry, DataDog)
- [ ] Configure auto-scaling
- [ ] Set up database backups

## Files

```
api/
├── manage.py                 # Django management
├── config/
│   ├── settings.py          # Django settings
│   ├── urls.py              # URL routing
│   └── wsgi.py              # WSGI entry point
├── analyzer/
│   ├── views.py             # REST API views
│   ├── models.py            # Database models
│   ├── serializers.py       # DRF serializers
│   ├── inference.py         # Model inference engine
│   └── apps.py              # App configuration
├── Dockerfile               # Docker containerization
├── requirements.txt         # Python dependencies
├── run_server.sh            # Production run script
├── benchmark.py             # Performance testing
├── API_DOCS.md             # Full API documentation
└── README.md               # This file
```

## Confidence Gating

**Measured 2026-09-22** via `measure_confidence_impact.py` against the real trained classifier (2,330-clause held-out test set):

```
Confidently answered (both thresholds): 2,209 / 2,330 (94.8%)
Wrong among those:                      65 (2.9% error rate)
Routed to 'uncertain' (coverage cost):   121 (5.2%)
Reduction from adding the margin check:  0.0%
```

**The margin check currently does nothing** — this isn't a rounding artifact, it's a mathematical property of the current settings. With a binary classifier, `margin = 2 × confidence - 1`, so any clause clearing the `confidence ≥ 0.65` bar automatically has `margin ≥ 0.30`, already above the `min_margin = 0.15` threshold. The margin condition can never be the deciding factor at these values. Tracked as [LEX-13](https://linear.app/lakshya-mehta/issue/LEX-13/confidence-gates-margin-check-is-mathematically-dead-code) — fixing it means raising `min_margin` above 0.30, or removing the check as dead weight if it's not worth tuning further.

## Troubleshooting

### Models not loading
```bash
python manage.py shell
from analyzer.inference import ModelCache
ModelCache()
```

### High latency
- Check CPU usage
- Increase worker count
- Enable caching
- Monitor memory

### Connection errors
- Check if server is running
- Verify port 8000 is open
- Check Nginx configuration

## Contributing

This API is designed to be:
- **Correct** - confidence gating over guessing, tests before claims
- **Maintainable** - clean code, documentation that matches measured reality, not aspiration

## License

MIT
