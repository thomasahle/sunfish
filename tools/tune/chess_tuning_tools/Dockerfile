FROM python:3.11-slim

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
