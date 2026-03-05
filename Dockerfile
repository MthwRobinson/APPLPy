FROM cgr.dev/chainguard/python:latest-dev AS builder

WORKDIR /src
COPY . /src

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip wheel . -w /src/wheels

FROM cgr.dev/chainguard/python:latest-dev AS runtime

USER root

COPY --from=builder /src/wheels /tmp/wheels
RUN python3 -m pip install --no-index --find-links /tmp/wheels APPLPy \
    && rm -rf /tmp/wheels

WORKDIR /app
CMD ["python3"]
