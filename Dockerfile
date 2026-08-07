FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard]

# Copy application code (data files excluded via .dockerignore)
COPY *.py ./
COPY sydney_geography.py ./
COPY inside_airbnb_*.py ./
COPY prepare_*.py ./
COPY premium_listing_features.py ./
COPY config/ ./config/
COPY artifacts/ ./artifacts/
COPY examples/ ./examples/

# Create directories for runtime outputs
RUN mkdir -p predictions reports

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "inside_airbnb_serve:app", "--host", "0.0.0.0", "--port", "8000"]
