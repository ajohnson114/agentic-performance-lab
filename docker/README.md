# Linux dev container

A development image for iterating on the Linux-only isolation layer
(`perflab/tools/isolation.py`, `perflab/tools/seccomp.py`) from a macOS box
without a push-and-wait CI loop. It is **not** a runtime for perflab itself —
DESIGN.md's "no container orchestration" decision covers candidate execution,
which still happens via bwrap directly on the host perflab runs on.

## Build

```sh
docker build -t perflab-linux-dev -f docker/Dockerfile .
```

## Run

bwrap creates user/mount namespaces, which Docker's default seccomp and
AppArmor profiles block — so the sandbox-under-test needs the outer sandbox
relaxed (fine on your own machine, don't do this with untrusted images):

```sh
docker run --rm -it \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --cap-add SYS_ADMIN \
  -v "$PWD":/work \
  perflab-linux-dev
```

If bwrap still can't create a sandbox on your Docker setup, fall back to
`--privileged` in place of the three security flags.

The image runs as the unprivileged `ubuntu` user (bwrap's root code path
tries to configure loopback inside `--unshare-net` and fails in a container);
don't override with `-u root` if you want the bwrap tests to pass.

The repo is mounted over `/work` (where the image did its editable install),
so code edits on the host are live inside the container — rebuild only when
dependencies change. Inside:

```sh
pytest tests/test_seccomp.py tests/test_isolation.py -q   # isolation + seccomp, incl. bwrap acceptance
pytest -q                                                 # full suite
```

## What this can and can't validate

- **Can**: the seccomp BPF filter for `--isolation=strict` (real kernel
  enforcement), bwrap acceptance tests, `auto` level resolution, any
  Linux-only code path. On Apple Silicon the container is arm64, so this
  exercises the aarch64 syscall table; CI's x86_64 runners cover the other
  table.
- **Can't**: anything needing PMU hardware (`perf stat` hardware counters,
  TMA, RAPL) — Docker Desktop's VM doesn't pass counters through — and any
  real benchmark timing (VM noise). For those, use a cloud Linux box
  (`setup-h100.sh`, `runpod-test.md`).
