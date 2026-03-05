---
name: dockerize-applpy
description: Containerize APPLPy using Chainguard base images from cgr.dev/chainguard.
    Trigger this skill when the user needs to update or review the Dockerfile.
---

# Wolfi Docker workflow

1. Update the Dockerfile based on the description in the user prompt.
2. Dockerfiles use the Chainguard wolfi-base base image. Packages in the image are managed with alpine package manager (apk). You are using Wolfi’s official package catalog (see https://github.com/wolfi-dev/os).
3. Use `make docker-build` to test that the docker image  buils successfully.
4. Use `grype` to test the image for vulnerabilities after building. If possible, resolved the vulnerabilities.
