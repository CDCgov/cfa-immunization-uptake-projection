FROM ubuntu:24.04

RUN apt update -y && apt install git -y

# Python from https://docs.astral.sh/uv/guides/integration/docker/
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Some handy uv environment variables
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_CACHE_DIR=/root/.cache/uv/python

RUN mkdir /cfa-immunization-uptake-projection
WORKDIR /cfa-immunization-uptake-projection

#
# Bring in python project dependency information and set the virtual env
#

# Dependency information
COPY pyproject.toml ./pyproject.toml
COPY uv.lock ./uv.lock

# Set VIRTUAL_ENV variable at runtime
ENV VIRTUAL_ENV=/cfa-immunization-uptake-projection/.venv

# Create the virtual environment
RUN uv venv "${VIRTUAL_ENV}"

# Update PATH to use the selected venv at runtime
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# Sync all python dependencies (excluding the local project itself)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

#
# Copy in python pipeline and orchestration files that frequently change
#

# Project files
COPY scripts .
COPY iup .
COPY tests .
COPY data/get_nis.py .
COPY README.md ./README.md

# Dagster
COPY dagster_defs.py ./dagster_defs.py