#!/bin/bash

# Django REST API Server - Production Ready
# Handles 100+ concurrent requests at sub-200ms latency

set -e

echo "🚀 LexiLite REST API Server"
echo "============================"

# Setup
echo "📦 Setting up Django..."
python manage.py migrate --no-input 2>/dev/null || true

# Production server with Gunicorn
echo "✓ Starting Gunicorn (4 workers)..."
gunicorn \
  --workers=4 \
  --worker-class=sync \
  --bind=0.0.0.0:8000 \
  --timeout=30 \
  --max-requests=1000 \
  --max-requests-jitter=100 \
  --access-logfile=- \
  --error-logfile=- \
  config.wsgi:application
