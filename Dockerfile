FROM cgr.dev/chainguard/python:latest-dev AS builder

USER nonroot

WORKDIR /src
COPY . /src

RUN uv sync --extra dev

ENTRYPOINT ["/usr/bin/uv", "run", "python"]

# CMD ["/bin/sh"]

# RUN python3 -m pip install --upgrade pip setuptools wheel
# RUN python3 -m pip wheel . -w /src/wheels
# RUN python3 -m pip install --no-index --find-links /src/wheels APPLPy --root /src/install
#
# FROM cgr.dev/chainguard/python:latest-dev AS runtime
#
# USER root
#
# COPY --from=builder /src/install/ /
#
# WORKDIR /app
# USER nonroot
# CMD ["python3"]
