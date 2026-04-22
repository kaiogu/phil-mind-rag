FROM python:3.12-slim

# System deps required by unstructured[pdf]
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency installation
RUN pip install --no-cache-dir uv

# Copy dependency specs first for layer caching and lockfile fidelity
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

# Copy project source
COPY phil_mind_rag/ ./phil_mind_rag/
COPY main.py ./

# Gradio listens on 0.0.0.0:7860 by default; HF Spaces expects port 7860
EXPOSE 7860

CMD ["uv", "run", "--frozen", "--no-dev", "python", "main.py"]
