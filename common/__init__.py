"""Benchmark-agnostic code shared by every experiment.

    paths.py    on-disk layout for results and derived products, and the .env loader
    models.py   model registry (family, litellm id, API-key env) and valid effort levels
    pricing.py  per-token rates and cost from a token breakdown
    usage.py    token accounting recovered from trajectories
    latency.py  output throughput and the waiting-time cost model
    anthropic_native_textbased.py
                mini-swe-agent model class that keeps Claude's thinking-token count
    sanitized_docker.py, sealed_docker.py
                container environments for the corrected SWE-Bench Pro setting

Entry-point scripts put the repository root on sys.path before importing this package:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import models, pricing
"""
