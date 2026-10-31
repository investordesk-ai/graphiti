FROM python:3.14.8-slim-bookworm@sha256:44e000a07b48bef1517a2b14adc93c17af728ca98858b5e24c6e1b18de1b2ef2

WORKDIR /app
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md /tmp/graphiti/
COPY graphiti_core /tmp/graphiti/graphiti_core/
RUN python -m pip install --no-cache-dir \
    "/tmp/graphiti[tracing]" pytest==9.1.1 pytest-asyncio==1.3.0 \
    pydantic==2.14.2 neo4j==6.4.1 openai==3.41.0 tenacity==9.2.1 \
    numpy==2.5.4 python-dotenv==1.2.4 posthog==7.81.0 httpx==0.28.1 \
    opentelemetry-api==1.45.2 opentelemetry-sdk==1.45.2 \
    && rm -rf /tmp/graphiti
