# syntax=docker/dockerfile:1

# ---- Chess Review web app -------------------------------------------------
# Self-contained image: bundles the Stockfish engine + the Polyglot opening
# book, so external users only open a URL — no local install required.
FROM python:3.12-slim

# Stockfish engine. The Debian package installs the binary at
# /usr/games/stockfish (not always on gunicorn's PATH, so we also point
# CHESS_ENGINE_PATH straight at it — engine.py honours that env var).
RUN apt-get update \
    && apt-get install -y --no-install-recommends stockfish \
    && rm -rf /var/lib/apt/lists/*

ENV CHESS_ENGINE_PATH=/usr/games/stockfish \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Copy metadata + source, then install the package (+ gunicorn for serving).
# package-data pulls in templates, the opening book (data/*.bin) and web assets.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . gunicorn

EXPOSE 8080

# Analysis is synchronous and CPU-heavy (Stockfish per move), so keep the
# worker count low and the request timeout generous.
CMD ["sh", "-c", "gunicorn 'chess_review.webapp:create_app()' --bind 0.0.0.0:${PORT} --workers 2 --threads 1 --timeout 300"]
