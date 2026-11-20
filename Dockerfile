FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY evaluation ./evaluation
RUN pip install --no-cache-dir ".[etl,rag,api]"

ENV INGENIUM_ROOT=/app
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
