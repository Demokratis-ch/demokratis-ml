FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ARG DOCKER_IMAGE_TAG
ENV DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG}

RUN useradd --create-home appuser
RUN mkdir /app && chown appuser:appuser /app
WORKDIR /app

ENV UV_SYSTEM_PYTHON=1
ENV PATH="/root/.local/bin:$PATH"

# Copy only the dependencies first to leverage Docker cache
COPY --chown=appuser:appuser pyproject.toml ../../uv.lock ./

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Copy the code
COPY --chown=appuser:appuser demokratis_ml demokratis_ml

USER appuser

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []
