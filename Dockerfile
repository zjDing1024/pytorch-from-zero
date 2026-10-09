# Metadata-pinned Python Official Image. Container execution is verified separately in CI.
ARG PYTHON_IMAGE=python:3.12.14-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254
FROM ${PYTHON_IMAGE} AS wheels
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY requirements/ requirements/
# Separate official indexes; no extra-index dependency ambiguity.
RUN python -m pip download --only-binary=:all: --no-deps --require-hashes \
      --index-url https://pypi.org/simple -r requirements/runtime-common.lock -d /wheels \
 && python -m pip download --only-binary=:all: --no-deps --require-hashes \
      --index-url https://download.pytorch.org/whl/cpu -r requirements/torch-cpu.lock -d /wheels
RUN --network=none python -m pip install --no-index --find-links=/wheels \
      --only-binary=:all: --require-hashes -r requirements/runtime-common.lock
COPY pyproject.toml README.md ./
COPY src/ src/
RUN --network=none python -m pip wheel --no-deps --no-build-isolation --no-index . -w /project

FROM ${PYTHON_IMAGE} AS runtime
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
COPY --from=wheels /wheels /wheels
COPY --from=wheels /project /project
COPY requirements/ /requirements/
RUN --network=none python -m pip install --no-index --find-links=/wheels \
      --only-binary=:all: --require-hashes -r /requirements/runtime-cpu.lock \
 && python -m pip install --no-index --no-deps /project/*.whl \
 && python -m pip check \
 && rm -rf /wheels /project
COPY scripts/check_installed.py /checks/check_installed.py
COPY THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
USER 10001:10001
WORKDIR /tmp
ENTRYPOINT ["python", "-I"]
CMD ["-m", "pytorch_lab.runtime_probe"]
