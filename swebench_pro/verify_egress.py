#!/usr/bin/env python3
"""SWE-Bench Pro-Corrected: check the network seal from inside a task container.

Spends no tokens. Brings up the proxy and network a sealed run uses
(`common.sealed_docker.ensure_proxy`), starts a task container on that network as the sweep would,
and runs probes from inside it.

Must fail (the code host is unreachable, including the bypass routes):
  * https to github.com, raw.githubusercontent.com and api.github.com; `git ls-remote`/`git fetch`;
  * a direct connection to a raw IP with the proxy disabled, a raw Python socket, external DNS,
    and curl with the proxy variables unset. These fail on routing alone because the network is
    `--internal`, so the seal does not depend on the agent honouring HTTP_PROXY.

Must succeed (the environment still works):
  * an allowed registry answers (pypi.org);
  * localhost is not sent through the proxy (the images' pip mirror is on 127.0.0.1:9876);
  * /app is intact and git works locally.

The seal is a property of the network, so one instance is normally enough.

Usage (needs Docker):
    python verify_egress.py
    python verify_egress.py --limit 3 --show-log      # also print what the proxy allowed/refused
    python verify_egress.py --teardown                # remove the proxy container and network

Writes `<output>/swebench_pro_corrected/egress_check.json`. verify_sanitization.py checks the
git-history strip.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, for `common`
from common import paths, sealed_docker

import variants

INSTANCES = paths.DATASETS_ROOT / "swebench_pro" / "mini_instances.jsonl"
DATASET = variants.resolve("pro-corrected")

# (id, must_succeed, shell), run inside the task container; a blocked attempt is a non-zero exit.
PROBES: list[tuple[str, bool, str]] = [
    # --- the code host ---
    ("github_https", False,
     "curl -sS --max-time 25 -o /dev/null https://github.com/"),
    ("github_raw", False,
     "curl -sS --max-time 25 -o /dev/null https://raw.githubusercontent.com/torvalds/linux/master/README"),
    ("github_api", False,
     "curl -sS --max-time 25 -o /dev/null https://api.github.com/repos/torvalds/linux/commits"),
    ("git_ls_remote", False,
     "git ls-remote https://github.com/torvalds/linux.git HEAD"),
    ("git_fetch", False,
     "cd /app && git fetch --quiet https://github.com/torvalds/linux.git master:refs/remotes/probe/x"),
    # --- bypass routes around the proxy ---
    ("bypass_direct_ip", False,
     "curl -sS --noproxy '*' --max-time 10 -o /dev/null https://140.82.121.4/"),
    ("bypass_raw_socket", False,
     "python3 -c \"import socket;socket.setdefaulttimeout(10);"
     "socket.create_connection(('140.82.121.4',443))\" || "
     "python -c \"import socket;socket.setdefaulttimeout(10);"
     "socket.create_connection(('140.82.121.4',443))\""),
    ("bypass_external_dns", False,
     "getent hosts github.com"),
    ("bypass_unset_proxy", False,
     "env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy "
     "curl -sS --max-time 15 -o /dev/null https://github.com/"),
    # --- the environment still works ---
    ("registry_pypi", True,
     "curl -sS --max-time 30 -o /dev/null https://pypi.org/simple/six/"),
    ("localhost_not_proxied", True,
     "curl -sS --max-time 10 -o /dev/null http://127.0.0.1:9876/ ; "
     "test $? -ne 5"),   # 5 = "couldn't resolve proxy" — the failure NO_PROXY prevents
    ("repo_intact", True,
     "cd /app && git rev-parse HEAD >/dev/null && test -n \"$(git ls-files | head -1)\""),
]


def load_instances(limit: int | None, filt: str) -> list[dict]:
    if not INSTANCES.exists():
        sys.exit(f"missing {INSTANCES}; run prepare_dataset.py first")
    rows = [json.loads(line) for line in INSTANCES.open()]
    if filt:
        rows = [r for r in rows if re.search(filt, r["instance_id"])]
    rows.sort(key=lambda r: r["instance_id"])
    if not rows:
        sys.exit("no instances selected")
    return rows[:limit or 1]


def merged_env_config() -> dict:
    """The environment block the sweep uses for the corrected dataset."""
    return variants.merged_config(DATASET).get("environment", {})


def docker(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def check_instance(instance: dict, env_cfg: dict, policy: dict) -> dict:
    iid = instance["instance_id"]
    row: dict = {"instance_id": iid, "ok": False, "probes": {}, "failures": []}
    env_flags = [f for k, v in env_cfg.get("env", {}).items() for f in ("-e", f"{k}={v}")]
    run_args = list(env_cfg.get("run_args", ["--rm"]))
    container = None
    try:
        created = docker("run", "-d", "-w", env_cfg.get("cwd", "/app"), *run_args,
                         "--network", env_cfg["proxy_network"], *env_flags,
                         instance["image_name"], "sleep", "30m", timeout=1800)
        if created.returncode != 0:
            row["failures"].append(f"could not start the container: {created.stderr.strip()[-300:]}")
            return row
        container = created.stdout.strip()

        for name, must_succeed, shell in PROBES:
            result = docker("exec", container, "bash", "-lc", shell, timeout=120)
            passed = (result.returncode == 0) if must_succeed else (result.returncode != 0)
            detail = (result.stderr or result.stdout).strip().splitlines()
            row["probes"][name] = {"expected": "allowed" if must_succeed else "blocked",
                                   "returncode": result.returncode, "as_expected": passed,
                                   "detail": detail[-1][:160] if detail else ""}
            if not passed:
                row["failures"].append(
                    f"{name}: expected {'success' if must_succeed else 'FAILURE'}, "
                    f"got returncode {result.returncode} — {(detail[-1][:160] if detail else '')}")
        row["ok"] = not row["failures"]
        return row
    except subprocess.TimeoutExpired as exc:
        row["failures"].append(f"timeout: {exc}")
        return row
    finally:
        if container:
            docker("rm", "-f", container, timeout=120)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=1, help="how many instances to probe (default 1: "
                                                         "the seal is a property of the network)")
    ap.add_argument("--filter", default="", help="regex on instance_id")
    ap.add_argument("--show-log", action="store_true",
                    help="also print what the proxy allowed and refused during this check")
    ap.add_argument("--teardown", action="store_true",
                    help="remove the proxy container and its network, then exit")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    env_cfg = merged_env_config()
    cfg = sealed_docker.SealedDockerEnvironmentConfig(image="unused", **{
        k: v for k, v in env_cfg.items()
        if k in sealed_docker.SealedDockerEnvironmentConfig.model_fields and k != "image"})

    if args.teardown:
        for line in sealed_docker.teardown(cfg) or ["nothing to remove"]:
            print(f"  {line}")
        return

    policy = sealed_docker.ensure_proxy(cfg)
    print(f"egress policy {policy['policy_sha256_16']} ({policy['n_allowed_hosts']} hosts allowed) "
          f"via {policy['proxy_image']} on network {policy['network']}")

    instances = load_instances(args.limit, args.filter)
    rows = []
    for instance in instances:
        row = check_instance(instance, env_cfg, policy)
        rows.append(row)
        mark = "ok  " if row["ok"] else "FAIL"
        print(f"  [{mark}] {row['instance_id'][:64]}")
        for name, probe in row["probes"].items():
            got = "blocked" if probe["returncode"] != 0 else "allowed"
            flag = " " if probe["as_expected"] else "<-- WRONG"
            print(f"         {name:22} want {probe['expected']:8} got {got:8}{flag}")
        for failure in row["failures"]:
            print(f"         {failure}")

    if args.show_log:
        log = docker("logs", "--tail", "60", cfg.proxy_container)
        print("\n--- proxy log (what it allowed and refused) ---")
        for line in (log.stdout + log.stderr).strip().splitlines()[-30:]:
            print(f"  {line}")

    n_ok = sum(r["ok"] for r in rows)
    print(f"\n{n_ok}/{len(rows)} instance(s) confirmed sealed")
    out = args.output or (paths.output_dir(paths.BENCHMARK_DATA_SUBDIR[DATASET.benchmark])
                          / "egress_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checked_utc": datetime.now(timezone.utc).isoformat(),
                               "policy": policy, "n_ok": n_ok, "n_instances": len(rows),
                               "rows": rows}, indent=2))
    print(f"  wrote {out}")
    if n_ok != len(rows):
        print("\nSome probe did not behave as expected: the network is not sealed as "
              "SWE-Bench Pro-Corrected requires.")
        sys.exit(1)


if __name__ == "__main__":
    main()
