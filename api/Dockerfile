# Build from the REPO ROOT, not api/, since the trained model files live in
# train_model/ as a sibling of api/ and both need to end up in the image:
#   docker build -f api/Dockerfile -t lexilite-api .
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ /app/
# Only the trained artifacts, not the training scripts/raw dataset — the
# running API needs the classifier and encoder, nothing else from
# train_model/. Path mirrors settings.TRAIN_MODEL_DIR's default (a sibling
# of api/), so no env override is needed to find it inside the container.
COPY train_model/saved_model /train_model/saved_model

EXPOSE 8000

# sync worker class: every view in this app (analyzer/views.py) is a plain
# synchronous Django/DRF view, not ASGI — an uvicorn (ASGI) worker class here
# would be running the wrong protocol for the app it's serving.
CMD ["gunicorn", "--workers=4", "--worker-class=sync", \
     "--bind=0.0.0.0:8000", "--timeout=30", "--max-requests=1000", \
     "--max-requests-jitter=100", "config.wsgi:application"]
