FROM cgr.dev/chainguard/python:latest-dev AS builder

USER nonroot

WORKDIR /src
COPY . /src

RUN uv sync --extra dev

ENTRYPOINT ["/usr/bin/uv", "run", "python"]
