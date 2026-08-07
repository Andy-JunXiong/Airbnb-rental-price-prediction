FROM python:3.12-slim

# Build argument for the artifact path inside the container.
# The artifact should be mounted at runtime, not baked into the image:
#   docker run -v $(pwd)/artifacts:/models -e MODEL_ARTIFACT_PATH=/models/inside_airbnb_quote_mvp.joblib ...
ARG MODEL_ARTIFACT_PATH=/models/inside_airbnb_quote_mvp.joblib
ENV MODEL_ARTIFACT_PATH=${MODEL_ARTIFACT_PATH}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard]

COPY *.py ./
COPY config/ ./config/
COPY dashboard.html ./
COPY examples/ ./examples/

RUN mkdir -p /models predictions reports

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
    CMD python -c "import urllib.request; \
    resp = urllib.request.urlopen('http://localhost:8000/health'); \
    import json; data = json.loads(resp.read()); \
    exit(0 if data.get('artifact_loaded') else 1)"

CMD ["python", "-m", "uvicorn", "inside_airbnb_serve:app", "--host", "0.0.0.0", "--port", "8000"]
