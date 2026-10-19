# LexiLite REST API

Production-ready Django REST API for real-time legal document analysis.

## Features

✅ **Real-time analysis** - Clause risk classification via Sentence-BERT + logistic regression
✅ **Batch processing** - Efficient model encoding
✅ **Authenticated** - session login for the browser page, API tokens for scripts — every user only sees their own analysis history (see [API_DOCS.md](API_DOCS.md#authentication))
⚠️ **Confidence gating** - implemented and measured (`analyzer/confidence.py`), but **not yet applied by the live endpoint**: every response reports each clause's confidence, but nothing is routed to review yet (see [Confidence gating](#confidence-gating) below)
✅ **Performance monitoring** - Built-in metrics & health checks
✅ **PDF/DOCX upload** - Not just raw text
✅ **Data retention** - analyses auto-expire after 30 days and users can delete their own data (see [DATA_POLICY.md](DATA_POLICY.md))
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
| DELETE | `/api/documents/<id>/` | Delete one of your analyses |
| DELETE | `/api/documents/delete-all/` | Delete all of your analyses |
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

**Status: built and measured, not yet wired into `/analyze`.** The live API returns each risky clause with its `confidence`, but does not mark low-confidence clauses `uncertain` or route them anywhere. Wiring it in is tracked as [LEX-16](https://linear.app/lakshya-mehta/issue/LEX-16).

**Measured 2026-10-19** with `python measure_confidence_impact.py` (run on the PC, RTX 5060 Ti) against the real trained classifier, on the 2,330-clause held-out test set (324 truly risky):

```
min_conf  answered  coverage  error (answered)  uncertain  risky recall
    0.50      2330   100.0%             5.11%          0        74.1%   <- no gate (plain argmax)
    0.55      2296    98.5%             4.66%         34        71.3%
    0.60      2251    96.6%             3.78%         79        67.3%
    0.65      2209    94.8%             2.94%        121        64.8%   <- current default
    0.70      2157    92.6%             2.50%        173        61.1%
    0.75      2108    90.5%             1.94%        222        56.2%
    0.80      2035    87.3%             1.33%        295        51.5%
    0.85      1949    83.6%             0.82%        381        43.5%
    0.90      1808    77.6%             0.44%        522        34.0%
    0.95      1540    66.1%             0.32%        790        21.9%
```

How to read it: at the default 0.65 the model answers 94.8% of clauses and its error rate on those drops from 5.11% to 2.94% (a ~42% relative reduction). **The cost is real:** 121 clauses become `uncertain`, and only 64.8% of truly risky clauses are both answered and flagged risky (74.1% with no gate). For a tool whose job is finding risky clauses, that recall drop is the number to weigh, and `uncertain` clauses are only useful if something handles them (a secondary model, LEX-8, or a human).

**The margin check was removed from the defaults (LEX-13).** For a binary classifier `margin = 2 × confidence − 1`, so "margin ≥ m" is exactly "confidence ≥ (1 + m) / 2": it can never reject anything the confidence threshold doesn't already reject. `ConfidenceScorer(min_margin=...)` still exists, off by default, because it does carry independent information with 3+ classes. To make the gate stricter for this model, raise `min_confidence` (see the table).

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
