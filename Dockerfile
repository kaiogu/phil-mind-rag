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

# Copy dependency spec first for layer caching
COPY pyproject.toml ./
RUN uv pip install --system --no-cache-dir \
    "chromadb>=1.5.2" \
    "gradio>=6.8.0" \
    "openai>=2.24.0" \
    "pydantic-settings>=2.13.1" \
    "python-dotenv>=1.2.2" \
    "ragas>=0.4.3" \
    "sentence-transformers>=5.2.3" \
    "tiktoken>=0.12.0" \
    "unstructured[pdf]>=0.21.5"

# Copy project source
COPY phil_mind_rag/ ./phil_mind_rag/
COPY main.py ./

# Gradio listens on 0.0.0.0:7860 by default; HF Spaces expects port 7860
EXPOSE 7860

CMD ["python", "main.py"]
