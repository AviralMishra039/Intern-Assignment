FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-fetch the tokenizer vocabulary and reranker model so the first request
# (and cold starts) don't need to download them.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
 && python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='./.flashrank_cache')"

RUN mkdir -p uploads chroma_data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
