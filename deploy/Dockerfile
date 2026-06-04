# ---- Breast Ultrasound U-Net API ----
# Uses uv (https://github.com/astral-sh/uv) for fast, reproducible installs.
FROM python:3.11-slim

# Copy the standalone uv binary from the official image (no pip bootstrap needed).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# HF Spaces runs as a non-root user with UID 1000; match it so files are writable.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV HOME=/home/user

WORKDIR /app

# Install dependencies first (better layer caching). Only the project metadata
# is copied at this stage so code edits don't bust the dependency layer.
COPY --chown=user pyproject.toml ./
RUN uv sync --no-install-project

# Now copy the app source + weights.
COPY --chown=user . .

# uv created a venv at /app/.venv; put it on PATH.
ENV PATH="/app/.venv/bin:$PATH"

# HF Spaces (and our local runs) expose the app on port 7860.
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
