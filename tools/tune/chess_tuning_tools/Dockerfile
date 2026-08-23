FROM python:3.11-slim@sha256:9c900dea9e8fb7e16277c179b555cc72d29a352dbc33cff48ad5a0412fd5bfc7

ARG CTT_COMMIT=a88f94725ac074ba4df635e263ffa98a502cee08
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && pip install --no-cache-dir \
        "git+https://github.com/kiudee/chess-tuning-tools.git@${CTT_COMMIT}" \
    && apt-get purge -y --auto-remove git \
    && rm -rf /var/lib/apt/lists/*

COPY patch_ctt.py /tmp/patch_ctt.py
RUN python /tmp/patch_ctt.py && rm /tmp/patch_ctt.py
COPY fastchess_shim.py /usr/local/bin/cutechess-cli

ENTRYPOINT ["tune"]
