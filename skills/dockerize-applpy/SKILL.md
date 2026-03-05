name: dockerize-applpy
description: Containerize APPLPy on cgr.dev/chainguard/wolfi-base using multi-stage builds, curated Wolfi packages, and a prebuilt wheel so the final image is as slim as possible. Trigger this skill when the user needs to rewrite or review the Dockerfile, build a container based on Wolfi, or tune the image size/runtime dependencies of APPLPy.

# Wolfi Docker workflow

1. Start from `cgr.dev/chainguard/wolfi-base`, pass `PYTHON_VERSION` via `ARG`, and install the compiler/runtime packages with `wolfi pkg install --yes` so you are using Wolfi’s official package catalog (see https://github.com/wolfi-dev/os). Clean `/var/cache/apk/*` right after each install.
2. In the builder stage install `python${PYTHON_VERSION}`, `python${PYTHON_VERSION}-dev`, `python${PYTHON_VERSION}-pip`, and the build tools (`build-base`, `git`, OpenSSL, libffi, zlib, xz, bzip2, wget). Use `pip install --upgrade pip setuptools wheel` and `python -m pip wheel .` to produce wheels without pulling runtime dependencies into the final image.
3. In the final stage install only the runtime packages (`python`, `python-pip`, `ca-certificates`), copy the wheel folder from the builder, and run `pip install --no-index --find-links /tmp/wheels APPLPy`. Remove `/tmp/wheels` afterwards so nothing extra remains.
4. Keep the default working directory at `/app`, expose a simple `CMD ["python3"]` if the goal is to run REPL or scripts, or override it when a user requests a specific entrypoint.
5. Maintain a `.dockerignore` that excludes Git metadata, caches, egg-info/build/dist directories, docs/_build, temporary files, and virtual environments so the build context stays minimal.
6. When revising the Dockerfile for size or reproducibility, prefer multi-stage builds, avoid installing dev-only packages in the runtime image, and reuse wheel artifacts between builds.

