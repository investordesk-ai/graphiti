FROM ghcr.io/investordesk-ai/base-image:0.3.35-dev@sha256:a66b987929b4d7791342643648bb8f29896ed9270446e704785055278d7416fe
RUN /app/venv/bin/python -m pip install --no-cache-dir pytest==9.1.1 pytest-asyncio==1.3.0
