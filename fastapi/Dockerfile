FROM python:3.13.7-slim

# ctranslate2/onnxruntime (pulled in by faster-whisper) need OpenMP at runtime,
# which Debian's slim images strip out by default. libsndfile1 is a defensive
# fallback for soundfile's WAV decoding (its wheel usually bundles this already).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libsndfile1 \
        build-essential \
        cmake \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies in their own layer so Docker can reuse this layer on
# rebuilds where only application code changed, not requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + trained model artifacts (app/artifacts/*).
COPY . .

# Makes print()/logging output show up immediately in Railway's log viewer
# instead of being buffered.
ENV PYTHONUNBUFFERED=1

# Documentation only — Railway assigns the real port via $PORT at runtime.
EXPOSE 8000

# Shell form (not exec-array form) so $PORT is actually expanded at container
# start. No --reload in production.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
