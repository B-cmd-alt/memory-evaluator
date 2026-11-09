# Interactive evaluator service. CPU-only torch for the embedder and reranker; models are baked into
# the image so a cold start does not download anything.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf TOKENIZERS_PARALLELISM=false
WORKDIR /app
RUN pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
COPY tools ./tools
COPY app ./app
COPY site ./site
COPY results/s1_dataset.json results/sixaxis_S1_seed42.json ./results/
COPY results/feasibility ./results/feasibility
COPY results/judge_calibration ./results/judge_calibration
RUN pip install --quiet . google-cloud-storage \
 && python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
ENV RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 PORT=8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
