FROM python:3.10-slim-bookworm

WORKDIR /app

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources || true

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# The application explicitly runs embeddings and reranking on CPU. Installing
# PyTorch first from its CPU wheel index prevents the general PyPI resolver from
# pulling multi-gigabyte CUDA runtime packages through sentence-transformers.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Keep the project package importable even when Streamlit executes src/app.py
# as a script and changes the first entry on Python's module search path.
ENV PYTHONPATH=/app

# Copy the entire project (including the src folder)
COPY . .

EXPOSE 8501

# CRITICAL FIX: Point to src/app.py instead of just app.py
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
