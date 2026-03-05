ARG PYTHON_VERSION=3.12

FROM cgr.dev/chainguard/wolfi-base AS builder
ARG PYTHON_VERSION

RUN wolfi pkg install --yes \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-pip \
    build-base \
    git \
    openssl \
    libffi-dev \
    bzip2-dev \
    xz-dev \
    zlib-dev \
    wget \
    && rm -rf /var/cache/apk/*

WORKDIR /src
COPY . /src

RUN python${PYTHON_VERSION} -m pip install --upgrade pip setuptools wheel
RUN python${PYTHON_VERSION} -m pip wheel . -w /out

FROM cgr.dev/chainguard/wolfi-base AS runtime
ARG PYTHON_VERSION

RUN wolfi pkg install --yes \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-pip \
    ca-certificates \
    && rm -rf /var/cache/apk/*

COPY --from=builder /out /tmp/wheels
RUN python${PYTHON_VERSION} -m pip install --no-index --find-links /tmp/wheels APPLPy \
    && rm -rf /tmp/wheels

WORKDIR /app
CMD ["python3"]
