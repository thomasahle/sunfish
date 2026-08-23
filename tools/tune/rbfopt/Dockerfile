FROM python:3.11-slim@sha256:9c900dea9e8fb7e16277c179b555cc72d29a352dbc33cff48ad5a0412fd5bfc7

ARG RBFOPT_COMMIT=458402c6f0c1d57f93ef2caaba461a51054e00ea

RUN apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq coinor-ipopt git \
    && pip install --no-cache-dir "git+https://github.com/coin-or/rbfopt.git@${RBFOPT_COMMIT}" \
    && apt-get purge -y -qq git \
    && apt-get autoremove -y -qq \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["python", "/tune/run_rbfopt.py"]
