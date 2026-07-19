# Local Linux dev loop for the bwrap/seccomp isolation layer (perflab/tools/
# isolation.py + perflab/tools/seccomp.py). This is a DEVELOPMENT image, not a
# runtime: DESIGN.md's rejection of container orchestration is about how
# perflab executes candidates, not about how we test Linux-only code from a
# macOS dev box. See docker/README.md for build/run commands (bwrap inside
# Docker needs relaxed security options).
#
# Mirrors the CI ubuntu jobs: ubuntu 24.04, python 3.12 (24.04's default),
# bubblewrap, editable install. Mount the repo over /work at run time so the
# editable install picks up live edits without a rebuild.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        bubblewrap \
        g++ \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY . /work

# Deps live in /opt/venv (outside /work) so the runtime bind-mount of the repo
# doesn't shadow them. `-e .` leaves perflab importable from /work, which the
# mount then replaces with the live tree.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -e ".[all,tasks-python]" pytest mypy ruff

ENV PATH=/opt/venv/bin:$PATH

# Run unprivileged (24.04's stock `ubuntu` user, uid 1000), mirroring the CI
# runners. As root, bwrap takes its privileged path and tries to configure
# loopback inside --unshare-net, which fails in this container ("Failed
# RTM_NEWADDR"); as a normal user it creates a user namespace and skips that,
# exactly like production use.
USER ubuntu
ENV HOME=/home/ubuntu

CMD ["bash"]
