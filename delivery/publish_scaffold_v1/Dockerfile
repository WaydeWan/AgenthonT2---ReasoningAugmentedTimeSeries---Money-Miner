FROM python:3.13.12-slim-bookworm@sha256:a58daefb915e1e03ad48f3ca4df8832065412c5c35cacb9d39f4229184de12b6
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HOME=/tmp
# Same numerical dependencies as the existing tested base. No private index/token.
RUN pip install --no-cache-dir numpy==2.1.3 pandas==2.2.3 scipy==1.15.3 pyarrow==23.0.0 scikit-learn==1.6.1 threadpoolctl==3.6.0 PyYAML==6.0.3 jsonschema==4.26.0 pytest==8.4.2
COPY vendor/agenthon-toolkit-v2.6.0/common /opt/toolkit/common
COPY vendor/track2-60509df/pyproject.toml /opt/forecasting/pyproject.toml
COPY vendor/track2-60509df/qfbench2_track_forecasting /opt/forecasting/qfbench2_track_forecasting
COPY vendor/track2-60509df/LICENSE /opt/forecasting/LICENSE
COPY vendor/track2-60509df/THIRD-PARTY-NOTICES.md /opt/forecasting/THIRD-PARTY-NOTICES.md
RUN pip install --no-cache-dir --no-deps /opt/toolkit/common /opt/forecasting
LABEL qfbench2.interface_version="2.0"
LABEL org.opencontainers.image.source="https://github.com/WaydeWan/AgenthonT2---ReasoningAugmentedTimeSeries---Money-Miner"
COPY submission_candidates /app/submission_candidates
COPY submission_candidates/release_artifacts/ARTIFACT_PROVENANCE.md /app/ARTIFACT_PROVENANCE.md
COPY LICENSE /app/LICENSE
COPY release-config.json /app/release-config.json
COPY release-identity.json /app/release-identity.json
COPY forecast /usr/local/bin/forecast
RUN chmod 755 /usr/local/bin/forecast && chmod -R a+rX /app
USER 65534:65534
WORKDIR /app
ENTRYPOINT []
CMD ["forecast", "--help"]
