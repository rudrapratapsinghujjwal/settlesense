FROM python:3.11-slim

# Metadata for Hugging Face Spaces Docker SDK
LABEL maintainer="SettleSense"
LABEL description="SettleSense — AI Finance Controller for Razorpay Buildathon Track 04"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (security best practice)
RUN useradd -m -u 1000 settlesense
WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/raw data/tune data/validation data/holdout data/answer_keys data/razorpay_live

# Set ownership
RUN chown -R settlesense:settlesense /app
USER settlesense

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Environment defaults (secrets are injected via HF Spaces Settings)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=INFO
ENV LLM_PROVIDER=mock
ENV RANDOM_SEED=42
ENV CONFIDENCE_THRESHOLD=0.70


# Run Streamlit on port 7860 bound to all interfaces
# --server.fileWatcherType=none: disable file watching in production (saves resources)
CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]
