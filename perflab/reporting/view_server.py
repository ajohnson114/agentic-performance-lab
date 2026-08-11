"""Serve a run directory over loopback so speedscope.app can fetch its profiles.

The documented way to open a py-spy profile is to drag the JSON onto
speedscope.app. That drag fails whenever the browser cannot read the dropped
path, which on macOS is common and hard to diagnose: a Gatekeeper-translocated
editor hands over a path from a randomized read-only mount, a repo under a
TCC-protected directory (``~/Documents``, ``~/Desktop``) needs a sandbox
extension that the drag never carries, and a browser installed outside
``/Applications`` compounds both. The failure surfaces as a dead drop or an
"Aw, Snap!" tab, never as a permissions error.

Serving the run over ``http://127.0.0.1`` removes the filesystem from the
equation for anything that can fetch from here.

Note what this server deliberately does *not* try to do. Pointing the hosted
speedscope.app at these URLs does not work in current Chrome, which answers the
cross-origin fetch with::

    blocked by CORS policy: Permission was denied for this request to
    access the `loopback` address space.

That is Local Network Access, and it is a *user permission*, not a header
negotiation -- it supersedes the older ``Access-Control-Allow-Private-Network``
response header, which Chrome no longer honours. No header this server sends
can grant it, so none is sent. ``Access-Control-Allow-Origin`` is still emitted
because it is what makes the fetch legal once the permission is granted, and
because a same-origin viewer served from this port needs no permission at all.

An https page fetching ``http://127.0.0.1`` is separately *not* mixed content:
loopback is "potentially trustworthy" under the Secure Contexts spec. That
exemption is real but irrelevant while the permission gate is closed.
"""
from __future__ import annotations

import functools
import logging
import socketserver
import subprocess
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logger = logging.getLogger(__name__)

#: What py-spy writes and every profiler run copies into its artifacts dir.
PROFILE_FILENAME = "pyspy_speedscope.json"

#: Artifact directory -> label, in display order. These sit beside dashboard.html
#: in a run directory, so the directory name doubles as the served URL prefix.
_ARTIFACT_DIRS = (("artifacts", "optimized"), ("artifacts_baseline", "baseline"))


@dataclass(frozen=True)
class Profile:
    """A speedscope profile found inside a run directory."""

    label: str
    path: Path
    #: Path relative to the run directory — also its URL path once served.
    rel: str


def find_profiles(run_dir: Path) -> list[Profile]:
    """Return the speedscope profiles in *run_dir*, optimized first.

    Zero-byte files are skipped: a profiler killed mid-write leaves one behind,
    and handing that to speedscope produces a parse error rather than an
    empty-looking graph.
    """
    found: list[Profile] = []
    for dirname, label in _ARTIFACT_DIRS:
        path = run_dir / dirname / PROFILE_FILENAME
        try:
            usable = path.is_file() and path.stat().st_size > 0
        except OSError:
            usable = False
        if usable:
            found.append(Profile(label=label, path=path, rel=f"{dirname}/{PROFILE_FILENAME}"))
    return found


def reveal_in_file_manager(path: Path) -> bool:
    """Open the OS file manager with *path* selected. True if a command ran.

    This is the escape hatch for the case the browser cannot be talked into:
    dragging the profile onto speedscope.app from a *file manager* works, while
    dragging the same file out of a sandboxed or Gatekeeper-translocated editor
    is silently refused. Revealing it removes the only hard part of that route,
    which is finding a file buried under out/runs/<timestamp>-<hash>/artifacts/.
    """
    if sys.platform == "darwin":
        argv = ["open", "-R", str(path)]
    elif sys.platform == "win32":
        argv = ["explorer", f"/select,{path}"]
    else:
        # No portable "select this file" on Linux; opening the folder is close.
        argv = ["xdg-open", str(path.parent)]
    try:
        subprocess.run(argv, check=False)
    except (OSError, subprocess.SubprocessError):
        logger.debug("Could not reveal %s with %s", path, argv[0], exc_info=True)
        return False
    return True


class _CorsHandler(SimpleHTTPRequestHandler):
    """Static file handler that speedscope.app is permitted to fetch from."""

    def __init__(self, *args, quiet: bool = True, **kwargs) -> None:
        # Set before super().__init__, which handles the whole request inline
        # and may call log_message before returning.
        self._quiet = quiet
        super().__init__(*args, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        # A run directory is rewritten in place across iterations; a cached
        # profile would show the previous candidate's flame graph.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler naming
        """Answer the CORS preflight a cross-origin viewer sends first."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        if not self._quiet:
            logger.debug("%s - %s", self.address_string(), format % args)


class _LoopbackServer(ThreadingHTTPServer):
    """A ThreadingHTTPServer that does not reverse-DNS its own address on bind.

    ``HTTPServer.server_bind`` calls ``socket.getfqdn(host)``, which issues a
    reverse DNS lookup for 127.0.0.1. Whenever the resolver cannot answer that
    -- VPN, captive portal, offline laptop -- it blocks until the lookup times
    out, measured here at 35 seconds before the first byte is served. The value
    only ever feeds CGI-style variables this handler never emits, so using the
    literal address costs nothing and makes startup instant.
    """

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[0], self.server_address[1]
        self.server_name = str(host)
        self.server_port = int(port)


def make_server(run_dir: Path, port: int = 0, quiet: bool = True) -> ThreadingHTTPServer:
    """Bind a loopback-only static server rooted at *run_dir*.

    Binds 127.0.0.1 rather than every interface on purpose: a run directory
    holds source paths, timings, LLM responses and generated candidate code,
    none of which should become readable to the local network just because
    someone opened a flame graph. Port 0 lets the OS pick a free port.

    ``SimpleHTTPRequestHandler`` resolves request paths against *run_dir* and
    rejects traversal out of it, so the served surface is exactly this run.
    """
    handler = functools.partial(_CorsHandler, directory=str(run_dir), quiet=quiet)
    server = _LoopbackServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server


def base_url(server: ThreadingHTTPServer) -> str:
    """Return the ``http://host:port`` root a bound *server* is listening on.

    Reads ``server_name``/``server_port`` rather than indexing
    ``server_address``, which typeshed widens to include the AF_UNIX bytes form
    that this AF_INET-only server can never produce.
    """
    return f"http://{server.server_name}:{server.server_port}"
