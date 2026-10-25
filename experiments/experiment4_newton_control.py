"""
Experiment 4A: Newton-control Lyapunov-exponent benchmark.

Control problem from the convergence-order paper:
    f(x) = (x - 2)^2

Newton update:
    U(x) = x - f(x)/f'(x) = (x + 2)/2

Hence:
    L = 2
    U'(L) = 1/2
    lambda_true = log(1/2) = -log(2)

Paper-aligned settings:
- 10 random initial points around 7.0e9 +/- 7.0e7
- K in {10, 20}

Extended settings for consistency with earlier LE experiments:
- K in {40, 50}

Run from repository root:
    python -m experiments.experiment4_newton_control
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import time
from pathlib import Path

import mpmath as mp
import numpy as np

from src.estimators.proposed import proposed_le_estimator
from src.estimators.yang_wu import yang_wu_scalar_le_estimator
from src.estimators.sun_wu import sun_wu_scalar_rbf_le_estimator
from src.estimators.rosenstein import rosenstein_le_estimator


def generate_trajectory(x0, steps):
    L = mp.mpf("2")
    x = [mp.mpf(x0)]
    for _ in range(steps + 1):
        x.append((x[-1] + L) / 2)
    return x


def initial_points(seed, replicates):
    rng = random.Random(seed)
    center = mp.mpf("7.0e9")
    half_width = mp.mpf("7.0e7")
    return [
        center + half_width * mp.mpf(str(rng.uniform(-1.0, 1.0)))
        for _ in range(replicates)
    ]


def estimate(data, K, proposed_only=False):
    methods = [
        ("Proposed-WB", lambda: proposed_le_estimator(
            data, K=K, end_index=K, limit_method="next",
            numeric_backend="mpmath")),
        ("Proposed-Aitken", lambda: proposed_le_estimator(
            data, K=K, end_index=K, limit_method="aitken",
            numeric_backend="mpmath")),
    ]

    if not proposed_only:
        methods += [
            ("Yang-Wu", lambda: yang_wu_scalar_le_estimator(
                data, evolution=1, r_min=None, numeric_backend="mpmath")),
            ("Sun-Wu", lambda: sun_wu_scalar_rbf_le_estimator(
                data, max_hidden_nodes=20, width_neighbors=None)),
            ("Rosenstein", lambda: rosenstein_le_estimator(
                data, embedding_dim=1, lag=1, temporal_exclusion=0,
                sample_time=mp.mpf(1), max_horizon=None,
                numeric_backend="mpmath")),
        ]

    for name, fn in methods:
        started = time.perf_counter()
        try:
            result = fn()
            value = result.get("lambda_hat", mp.nan)
            try:
                finite = bool(mp.isfinite(value))
            except Exception:
                finite = math.isfinite(float(value))
            valid = bool(result.get("valid", False) and finite)
            reason = result.get("failure_reason", "")
            if not valid and not reason:
                reason = "Estimator returned an invalid or nonfinite result."
        except Exception as exc:
            result, value, valid = {}, mp.nan, False
            reason = f"{type(exc).__name__}: {exc}"
        yield name, result, value, valid, reason, time.perf_counter() - started


def text_value(v):
    return mp.nstr(v, 70) if isinstance(v, mp.mpf) else v


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{k: text_value(v) for k, v in r.items()} for r in rows])


def median(values):
    values = sorted(values)
    n = len(values)
    if not n:
        return mp.nan
    return values[n//2] if n % 2 else (values[n//2-1] + values[n//2]) / 2


def summarize(rows):
    out = []
    for K, method in sorted({(r["K"], r["estimator"]) for r in rows}):
        g = [r for r in rows if r["K"] == K and r["estimator"] == method]
        v = [r for r in g if r["valid"]]
        e = [r["absolute_error"] for r in v]
        out.append({
            "K": K,
            "paper_aligned_K": K in (10, 20),
            "estimator": method,
            "total": len(g),
            "valid": len(v),
            "invalid": len(g)-len(v),
            "valid_rate": mp.mpf(len(v))/len(g),
            "mean_absolute_error": mp.fsum(e)/len(e) if e else mp.nan,
            "median_absolute_error": median(e),
            "max_absolute_error": max(e) if e else mp.nan,
            "correct_sign_fraction": mp.mpf(sum(r["correct_sign"] for r in g))/len(g),
        })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--windows", nargs="+", type=int, default=[10, 20, 40, 50])
    p.add_argument("--replicates", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dps", type=int, default=220)
    p.add_argument("--proposed-only", action="store_true")
    p.add_argument("--output", type=Path,
                   default=Path("results/experiment4_newton_control"))
    args = p.parse_args()

    if args.steps < 2 or min(args.windows) < 2 or max(args.windows) > args.steps:
        p.error("require steps >= 2 and 2 <= K <= steps")
    if args.replicates < 1:
        p.error("replicates must be >= 1")

    mp.mp.dps = args.dps
    args.output.mkdir(parents=True, exist_ok=True)

    L = mp.mpf("2")
    lam_true = mp.log(mp.mpf("0.5"))

    rows, diagnostics = [], []

    for rep, x0 in enumerate(initial_points(args.seed, args.replicates), start=1):
        x = generate_trajectory(x0, args.steps)

        rec_err = max(
            abs((x[t+1]-L) - mp.mpf("0.5")*(x[t]-L))
            for t in range(args.steps+1)
        )

        diagnostics.append({
            "replicate": rep,
            "x0": x0,
            "steps": args.steps,
            "fixed_point_true": L,
            "multiplier_true": mp.mpf("0.5"),
            "lambda_true": lam_true,
            "max_absolute_recurrence_error": rec_err,
            "final_absolute_error_to_fixed_point": abs(x[args.steps]-L),
        })

        for K in args.windows:
            n = args.steps
            data = x[n-K:n+2]

            for name, result, value, valid, reason, elapsed in estimate(
                data, K, args.proposed_only
            ):
                rows.append({
                    "replicate": rep,
                    "x0": x0,
                    "K": K,
                    "paper_aligned_K": K in (10, 20),
                    "end_index": n,
                    "window_start": n-K,
                    "latest_observation_index": n+1,
                    "estimator": name,
                    "lambda_true": lam_true,
                    "lambda_hat": value,
                    "absolute_error": abs(value-lam_true) if valid else mp.nan,
                    "correct_sign": bool(valid and value < 0),
                    "valid": valid,
                    "failure_reason": reason,
                    "L_hat": result.get("L_hat", mp.nan),
                    "limit_absolute_error":
                        abs(result["L_hat"]-L)
                        if valid and "L_hat" in result else mp.nan,
                    "ell_err_hat": result.get("ell_err_hat", mp.nan),
                    "id_fie_hat": result.get("id_fie_hat", mp.nan),
                    "runtime_sec": elapsed,
                })

        print(f"Completed replicate={rep:02d}, x0={mp.nstr(x0, 12)}", flush=True)

    summary = summarize(rows)

    write_csv(args.output/"estimates.csv", rows)
    write_csv(args.output/"trajectory_diagnostics.csv", diagnostics)
    write_csv(args.output/"final_summary.csv", summary)

    config = {
        "function": "(x-2)^2",
        "method": "Newton-Raphson",
        "newton_update": "(x+2)/2",
        "fixed_point_true": "2",
        "multiplier_true": "0.5",
        "lambda_true": mp.nstr(lam_true, 70),
        "paper_initial_point_rule": "7.0e9 +/- 7.0e7",
        "paper_replicates": 10,
        "paper_K_values": [10, 20],
        "extended_K_values": [K for K in args.windows if K not in (10, 20)],
        "distance_cutoff": None,
        "steps": args.steps,
        "windows": args.windows,
        "replicates": args.replicates,
        "seed": args.seed,
        "dps": args.dps,
        "proposed_only": args.proposed_only,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mpmath": mp.__version__,
        "timing": "single-call elapsed seconds; not a controlled runtime benchmark",
    }

    (args.output/"config.json").write_text(json.dumps(config, indent=2),
                                           encoding="utf-8")

    print("\n=== SUMMARY ===")
    for r in summary:
        print(
            f"K={r['K']:2d} {r['estimator']:<18} "
            f"valid={r['valid']}/{r['total']} "
            f"MAE={mp.nstr(r['mean_absolute_error'], 8)}"
        )


if __name__ == "__main__":
    main()
