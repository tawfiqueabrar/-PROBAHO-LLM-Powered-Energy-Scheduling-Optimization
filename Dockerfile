FROM python:3.11-slim

LABEL org.opencontainers.image.title="GridWise Energy Optimizer" \
      org.opencontainers.image.description="LLM-assisted smart campus energy scheduling API"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

# PuLP ships its CBC binary in the pinned wheel, so build tools are not needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "from urllib.request import urlopen; assert urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200" || exit 1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
