"""Tests for `perflab view` -- the loopback server that exposes a run directory.

Two of these encode findings that cost real debugging time and would silently
regress: the bind must not reverse-DNS (35s stall before serving a byte), and
the server must not advertise Private Network Access, a header current Chrome
ignores in favour of a user permission it alone can grant.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from perflab.reporting import view_server


def _make_run(tmp_path: Path, *, optimized=True, baseline=False, dashboard=True) -> Path:
    run_dir = tmp_path / "20260805-120000-abcd1234"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "meta.json").write_text('{"run_id": "20260805-120000-abcd1234"}')
    if optimized:
        (run_dir / "artifacts" / view_server.PROFILE_FILENAME).write_text(
            json.dumps({"profiles": [], "shared": {"frames": []}})
        )
    if baseline:
        (run_dir / "artifacts_baseline").mkdir()
        (run_dir / "artifacts_baseline" / view_server.PROFILE_FILENAME).write_text("{}")
    if dashboard:
        (run_dir / "dashboard.html").write_text("<h1>dash</h1>")
    return run_dir


@pytest.fixture
def served(tmp_path):
    """A running loopback server over a populated run directory."""
    run_dir = _make_run(tmp_path, baseline=True)
    server = view_server.make_server(run_dir)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield run_dir, view_server.base_url(server)
    finally:
        server.shutdown()
        server.server_close()


class TestRevealInFileManager:
    """Dragging from a file manager is the route that survives every sandbox."""

    def test_reveals_with_the_platform_command(self, tmp_path, monkeypatch):
        run_dir = _make_run(tmp_path)
        profile = view_server.find_profiles(run_dir)[0].path
        calls: list[list[str]] = []
        monkeypatch.setattr(view_server.subprocess, "run", lambda argv, **kw: calls.append(argv))
        monkeypatch.setattr(view_server.sys, "platform", "darwin")

        assert view_server.reveal_in_file_manager(profile) is True
        # -R selects the file rather than opening it in a JSON viewer.
        assert calls == [["open", "-R", str(profile)]]

    def test_linux_opens_the_containing_folder(self, tmp_path, monkeypatch):
        run_dir = _make_run(tmp_path)
        profile = view_server.find_profiles(run_dir)[0].path
        calls: list[list[str]] = []
        monkeypatch.setattr(view_server.subprocess, "run", lambda argv, **kw: calls.append(argv))
        monkeypatch.setattr(view_server.sys, "platform", "linux")

        assert view_server.reveal_in_file_manager(profile) is True
        assert calls == [["xdg-open", str(profile.parent)]]

    def test_missing_file_manager_is_not_fatal(self, tmp_path, monkeypatch):
        """A headless box has no file manager; `perflab view` must still serve."""
        run_dir = _make_run(tmp_path)
        profile = view_server.find_profiles(run_dir)[0].path

        def boom(*a, **k):
            raise FileNotFoundError("xdg-open")

        monkeypatch.setattr(view_server.subprocess, "run", boom)
        assert view_server.reveal_in_file_manager(profile) is False


class TestFindProfiles:
    def test_finds_both_optimized_and_baseline(self, tmp_path):
        profiles = view_server.find_profiles(_make_run(tmp_path, baseline=True))
        assert [p.label for p in profiles] == ["optimized", "baseline"]
        assert [p.rel for p in profiles] == [
            "artifacts/pyspy_speedscope.json",
            "artifacts_baseline/pyspy_speedscope.json",
        ]

    def test_optimized_is_first(self, tmp_path):
        """`perflab view` opens profiles[0]; that must be the optimized run."""
        assert view_server.find_profiles(_make_run(tmp_path, baseline=True))[0].label == "optimized"

    def test_missing_profiles_yield_empty(self, tmp_path):
        assert view_server.find_profiles(_make_run(tmp_path, optimized=False)) == []

    def test_zero_byte_profile_is_skipped(self, tmp_path):
        """A profiler killed mid-write leaves an empty file that cannot parse."""
        run_dir = _make_run(tmp_path)
        (run_dir / "artifacts" / view_server.PROFILE_FILENAME).write_text("")
        assert view_server.find_profiles(run_dir) == []

    def test_nonexistent_run_dir_is_not_an_error(self, tmp_path):
        assert view_server.find_profiles(tmp_path / "nope") == []


class TestCorsHeaders:
    def test_get_allows_cross_origin_reads(self, served):
        _, root = served
        with urllib.request.urlopen(f"{root}/artifacts/pyspy_speedscope.json") as resp:
            assert resp.headers["Access-Control-Allow-Origin"] == "*"

    def test_does_not_claim_private_network_access(self, served):
        """Chrome replaced that header with a user permission and ignores it now.

        Verified from a real console error: "Permission was denied for this
        request to access the `loopback` address space." Sending the header
        would only imply a guarantee this server cannot make.
        """
        _, root = served
        with urllib.request.urlopen(f"{root}/artifacts/pyspy_speedscope.json") as resp:
            assert resp.headers["Access-Control-Allow-Private-Network"] is None

    def test_preflight_is_answered(self, served):
        _, root = served
        req = urllib.request.Request(
            f"{root}/artifacts/pyspy_speedscope.json",
            method="OPTIONS",
            headers={
                "Origin": "https://www.speedscope.app",
                "Access-Control-Request-Method": "GET",
            },
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 204
            assert "GET" in resp.headers["Access-Control-Allow-Methods"]

    def test_profile_body_is_served_intact(self, served):
        run_dir, root = served
        with urllib.request.urlopen(f"{root}/artifacts/pyspy_speedscope.json") as resp:
            body = json.loads(resp.read())
        expected = json.loads((run_dir / "artifacts" / view_server.PROFILE_FILENAME).read_text())
        assert body == expected

    def test_responses_are_not_cached(self, served):
        """Run dirs are rewritten across iterations; a cached profile is stale."""
        _, root = served
        with urllib.request.urlopen(f"{root}/dashboard.html") as resp:
            assert resp.headers["Cache-Control"] == "no-store"


class TestServerBinding:
    def test_binds_loopback_only(self, tmp_path):
        """Run dirs hold source, timings and generated code -- never expose them."""
        server = view_server.make_server(_make_run(tmp_path))
        try:
            assert server.server_address[0] == "127.0.0.1"
        finally:
            server.server_close()

    def test_base_url_matches_bound_port(self, tmp_path):
        server = view_server.make_server(_make_run(tmp_path))
        try:
            assert view_server.base_url(server) == f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.server_close()

    def test_port_zero_picks_a_free_port(self, tmp_path):
        server = view_server.make_server(_make_run(tmp_path), port=0)
        try:
            assert server.server_address[1] > 0
        finally:
            server.server_close()

    def test_bind_does_not_reverse_dns(self, tmp_path, monkeypatch):
        """getfqdn() on 127.0.0.1 blocked 35s on a real machine before serving."""
        import socket as socket_mod

        def explode(*args, **kwargs):
            raise AssertionError("server_bind must not reverse-DNS the bind address")

        monkeypatch.setattr(socket_mod, "getfqdn", explode)
        server = view_server.make_server(_make_run(tmp_path))
        try:
            assert server.server_name == "127.0.0.1"
            assert server.server_port == server.server_address[1]
        finally:
            server.server_close()

    def test_traversal_outside_the_run_dir_is_refused(self, tmp_path):
        """The served surface is exactly one run, not the whole filesystem."""
        secret = tmp_path / "secret.txt"
        secret.write_text("PERFLAB_API_KEY=sk-ant-should-never-be-served")
        run_dir = _make_run(tmp_path)
        server = view_server.make_server(run_dir)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            root = view_server.base_url(server)
            for attack in ("/../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt"):
                body = ""
                try:
                    with urllib.request.urlopen(root + attack) as resp:
                        body = resp.read().decode()
                except urllib.error.HTTPError:
                    pass  # 404 is the expected outcome
                assert "sk-ant-should-never-be-served" not in body
        finally:
            server.shutdown()
            server.server_close()
