# syntax=docker/dockerfile:1
# Runtime image: API + configured trained artifact. Raw dataset is NOT included.
# Dependencies are expected as manylinux wheels (no apt/build-essential).

FROM python:3.12-slim-bookworm AS builder

ENV POETRY_VERSION=2.1.3 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    POETRY_REQUESTS_TIMEOUT=120

# Install pinned Poetry via pip (TLS verified). No curl/apt compilers —
# numpy/scipy/pandas/matplotlib ship wheels for CPython 3.12 on linux/amd64.
RUN set -eux; \
    ok=0; \
    for i in 1 2 3 4 5; do \
      if pip install --no-cache-dir "poetry==${POETRY_VERSION}"; then ok=1; break; fi; \
      echo "pip install poetry attempt $i failed; retrying in $((i * 20))s"; \
      sleep $((i * 20)); \
    done; \
    test "$ok" -eq 1

WORKDIR /app

COPY pyproject.toml poetry.lock ./
# Bounded retries for transient PyPI/TLS read timeouts (certificates remain verified).
RUN set -eux; \
    ok=0; \
    for i in 1 2 3 4 5; do \
      if poetry install --only main --no-root --no-ansi; then ok=1; break; fi; \
      echo "poetry install (deps) attempt $i failed; retrying in $((i * 20))s"; \
      sleep $((i * 20)); \
    done; \
    test "$ok" -eq 1

COPY README.md ./
COPY src ./src
COPY configs ./configs
# Install the project package only — dependency set already resolved above.
RUN poetry install --only-root --no-ansi


FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    AQUA_HOST=0.0.0.0 \
    AQUA_PORT=8000 \
    AQUA_ARTIFACT_PATH=/app/artifacts/models/llt_relative_health_v1.json

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin aqua \
    && mkdir -p /app \
    && chown -R aqua:aqua /app

WORKDIR /app

COPY --from=builder --chown=aqua:aqua /app/.venv /app/.venv
COPY --from=builder --chown=aqua:aqua /app/src /app/src
COPY --from=builder --chown=aqua:aqua /app/pyproject.toml /app/README.md /app/
COPY --chown=aqua:aqua configs /app/configs
COPY --chown=aqua:aqua artifacts/models/llt_relative_health_v1.json \
    /app/artifacts/models/llt_relative_health_v1.json
COPY --chown=aqua:aqua examples/forecast /app/examples/forecast

USER aqua

EXPOSE 8000

# Liveness only — readiness is GET /ready (artifact compatibility).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "aqua_fouling.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
