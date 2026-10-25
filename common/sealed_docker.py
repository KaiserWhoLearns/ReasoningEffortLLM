"""A task environment with the git history stripped and egress limited to package registries.

`SealedDockerEnvironment` extends `sanitized_docker.SetupCommandDockerEnvironment` (which runs the
history strip) with a network seal, following
https://cursor.com/blog/reward-hacking-coding-benchmarks: a pinned proxy allows dependency
resolution against an allow-list of package registries, and nothing else.

1. The task container joins an `--internal` Docker network, which has no route off the host:
   raw IPs, git://, ssh and plain sockets fail, and external DNS does not resolve.
2. A tinyproxy sidecar on that network and on `bridge` is the only exit. It refuses every
   destination host not in `egress_proxy/allowlist.txt`, filtering on the request host
   (including the `CONNECT host:443` of an HTTPS tunnel), so no TLS is intercepted.
   HTTP(S)_PROXY in the task container point at it; NO_PROXY keeps localhost direct, because the
   images pin pip to a mirror on 127.0.0.1:9876.

Because (1) is enforced by the network, unsetting HTTP_PROXY does not escape the seal.

One proxy and one network serve a whole sweep; `ensure_proxy()` is idempotent and thread-safe. The
proxy image is tagged with the policy sha (`policy()`, computed over the effective directives
only, not comments), so a running proxy built from a different allow-list is replaced. Each
trajectory and run manifest records the sha and the host list.

Check it without running an agent: `swebench_pro/verify_egress.py`.
"""
from __future__ import annotations

import hashlib
import pathlib
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from common.sanitized_docker import (SetupCommandDockerEnvironment,
                                     SetupCommandDockerEnvironmentConfig)

ASSETS = Path(__file__).resolve().parent / "egress_proxy"
ALLOWLIST = ASSETS / "allowlist.txt"

_LOCK = threading.Lock()
_READY: dict[str, dict] = {}   # policy tag -> policy record, so the checks run once per sweep


class SealedDockerEnvironmentConfig(SetupCommandDockerEnvironmentConfig):
    proxy_network: str = "cp_sealed_net"
    """Name of the `--internal` docker network the task containers join."""
    proxy_container: str = "cp-egress-proxy"
    """Name of the shared proxy container. Task containers reach it by this name (docker's
    embedded DNS resolves container names on a user-defined network)."""
    proxy_port: int = 8888
    proxy_image: str = "cp-egress-proxy"
    """Image repository; the tag is the policy sha256, so it is computed, not configured."""
    proxy_run_args: list[str] = []
    """Extra `docker run` flags for the proxy container (e.g. `--runtime runc`)."""
    proxy_build_timeout: int = 600


def _effective(path: pathlib.Path) -> list[str]:
    """A file's directives, with comments and blank lines dropped."""
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")]


def policy() -> dict:
    """The egress policy: the allowed hosts, and a sha of what the proxy enforces.

    The sha covers the effective directives of every file in egress_proxy/ (allowed hosts,
    tinyproxy config, Dockerfile), with comments and blank lines dropped, so editing a comment
    does not change the policy's identity.
    """
    digest = hashlib.sha256()
    for path in sorted(p for p in ASSETS.iterdir() if p.is_file()):
        digest.update(path.name.encode())
        digest.update("\n".join(_effective(path)).encode())
    hosts = _effective(ALLOWLIST)
    return {"sha256_16": digest.hexdigest()[:16], "allowed_hosts": hosts,
            "n_allowed_hosts": len(hosts)}


def _filter_file(hosts: list[str]) -> str:
    """tinyproxy filter: one POSIX-extended regex per allowed host, anchored.

    A leading dot in the allow-list means "and subdomains"; anything else must match exactly. Both
    are anchored at the end, so `github.com.evil.example` cannot pass as `github.com`.
    """
    lines = []
    for host in hosts:
        if host.startswith("."):
            lines.append(f"(^|\\.){host.lstrip('.').replace('.', chr(92) + '.')}$")
        else:
            lines.append(f"^{host.replace('.', chr(92) + '.')}$")
    return "\n".join(lines) + "\n"


def _docker(*args: str, timeout: int = 120, check: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(f"`docker {' '.join(args)}` failed ({result.returncode}): "
                           f"{(result.stderr or result.stdout).strip()[-500:]}")
    return result


def _image_exists(tag: str) -> bool:
    return _docker("image", "inspect", tag).returncode == 0


def _build_image(cfg: SealedDockerEnvironmentConfig, pol: dict, tag: str) -> None:
    """Build the proxy image with this policy's filter baked in.

    The filter is generated from `allowlist.txt` and baked into the image, so the image tag (the
    policy sha) determines what the proxy allows.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ctx = Path(tmp)
        for name in ("Dockerfile", "tinyproxy.conf"):
            shutil.copy(ASSETS / name, ctx / name)
        (ctx / "filter").write_text(_filter_file(pol["allowed_hosts"]))
        _docker("build", "-q", "-t", tag, str(ctx),
                timeout=cfg.proxy_build_timeout, check=True)


def _running_image(container: str) -> str | None:
    result = _docker("inspect", "-f", "{{.Config.Image}}", container)
    return result.stdout.strip() if result.returncode == 0 else None


def ensure_proxy(cfg: SealedDockerEnvironmentConfig) -> dict:
    """Bring up (or reuse) the internal network and the allow-list proxy. Idempotent, thread-safe.

    Returns the policy record that goes into the trajectory and the run manifest.
    """
    pol = policy()
    tag = f"{cfg.proxy_image}:{pol['sha256_16']}"
    with _LOCK:
        if tag in _READY:
            return _READY[tag]

        if not _image_exists(tag):
            _build_image(cfg, pol, tag)

        # `--internal` is the load-bearing flag: no NAT, no default route off the host.
        if _docker("network", "inspect", cfg.proxy_network).returncode != 0:
            _docker("network", "create", "--internal", cfg.proxy_network, check=True)

        # A proxy left over from a different policy would silently enforce the old allow-list.
        current = _running_image(cfg.proxy_container)
        if current is not None and current != tag:
            _docker("rm", "-f", cfg.proxy_container)
            current = None
        if current is None:
            _docker("run", "-d", "--name", cfg.proxy_container, "--restart", "unless-stopped",
                    *cfg.proxy_run_args, "--network", cfg.proxy_network, tag, check=True)
        elif _docker("inspect", "-f", "{{.State.Running}}",
                     cfg.proxy_container).stdout.strip() != "true":
            _docker("start", cfg.proxy_container, check=True)

        # The proxy, and only the proxy, also joins a network that reaches the internet.
        connect = _docker("network", "connect", "bridge", cfg.proxy_container)
        if connect.returncode != 0 and "already exists" not in connect.stderr:
            raise RuntimeError(f"could not give the proxy an outside leg: {connect.stderr.strip()}")

        record = {"proxy_image": tag, "policy_sha256_16": pol["sha256_16"],
                  "n_allowed_hosts": pol["n_allowed_hosts"],
                  "allowed_hosts": pol["allowed_hosts"],
                  "network": cfg.proxy_network, "container": cfg.proxy_container}
        _READY[tag] = record
        return record


def teardown(cfg: SealedDockerEnvironmentConfig | None = None) -> list[str]:
    """Remove the proxy and its network (not called automatically; the proxy is reused)."""
    cfg = cfg or SealedDockerEnvironmentConfig(image="unused")
    done = []
    if _docker("rm", "-f", cfg.proxy_container).returncode == 0:
        done.append(f"removed container {cfg.proxy_container}")
    if _docker("network", "rm", cfg.proxy_network).returncode == 0:
        done.append(f"removed network {cfg.proxy_network}")
    _READY.clear()
    return done


class SealedDockerEnvironment(SetupCommandDockerEnvironment):
    """Git history stripped (inherited) **and** egress restricted to the package allow-list."""

    def __init__(self, *, config_class: type = SealedDockerEnvironmentConfig, **kwargs):
        # The network is a `docker run` flag, so the config is parsed before the container starts.
        cfg = config_class(**kwargs)
        self.egress_policy = ensure_proxy(cfg)
        kwargs = {**kwargs, "run_args": [*cfg.run_args, "--network", cfg.proxy_network]}
        super().__init__(config_class=config_class, **kwargs)

    def serialize(self) -> dict:
        """Record the egress policy in the trajectory, next to the sanitizer's output."""
        data = super().serialize()
        data["info"]["egress_policy"] = self.egress_policy
        return data
