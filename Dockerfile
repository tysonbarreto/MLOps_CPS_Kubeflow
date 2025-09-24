FROM python:3.11-slim-bookworm AS compiler

WORKDIR /app
# install PDM
RUN pip install -U uv &&\
    uv venv --python 3.11

ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml pdm.lock README.md /app/
RUN uv sync
# disable update check
# ENV PDM_CHECK_UPDATE=false
# RUN pdm install

FROM python:3.11-slim-bookworm AS runner

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY --from=compiler /app/.venv/ /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
# copy files
COPY src/ /app/src
COPY templates/ /app/templates
COPY application.py /app/
COPY artifacts/ /app/artifacts
COPY Dockerfile /app/
#COPY mlops.json /app/
#ENV GOOGLE_APPLICATION_CREDENTIALS="/app/mlops.json"
#COPY dvc.yaml /app/

# Expose the application port
EXPOSE 8080

# RUN dvc init --no-scm 
# RUN dvc repro

RUN rm -rf /app/mlops.json
# set command/entrypoint, adapt to fit your needs
CMD ["python", "application.py"]