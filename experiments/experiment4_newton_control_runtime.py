"""
Experiment 4A runtime benchmark: Newton-control LE estimation.

Accuracy experiment:
    experiments/experiment4_newton_control.py

Control problem:
    f(x) = (x - 2)^2
    Newton update U(x) = (x + 2)/2
    L = 2
    lambda_true = log(1/2)

Runtime design
--------------
- Same 10 paper-aligned initial-condition replicates.
- K in {10, 20, 40, 50}.
- Exactly K+2 final observations are supplied to every estimator.
- The observed record is normalized in mpmath before float64 conversion:

      z_i = (x_i - x_last) / max_j |x_j - x_last|.

  This uses observed values only; the true fixed point and true LE are not used
  in the transformation.  The affine coordinate change preserves the scalar LE.
- Every estimator receives exactly the same float64 record.
- One warm-up call per estimator is performed outside timing.
- Trajectory generation and reference calculations are excluded from timing.
- Report median runtime as the primary empirical runtime statistic.

Run from repository root:

    python -m experiments.experiment4_newton_control_runtime
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path

import mpmath as mp
import numpy as np

from src.estimators.proposed import proposed_le_estimator
from src.estimators.yang_wu import yang_wu_scalar_le_estimator
from src.estimators.sun_wu import sun_wu_scalar_rbf_le_estimator
from src.estimators.rosenstein import rosenstein_le_estimator


METHODS = [
    "Proposed-WB",
    "Proposed-Aitken",
    "Yang-Wu",
    "Rosenstein",
    "Sun-Wu",
]


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
        center
        + half_width * mp.mpf(str(rng.uniform(-1.0, 1.0)))
        for _ in range(replicates)
    ]


def normalize_observed_record_to_float64(data):
    """
    Observed-only normalization performed before float64 conversion.

    The last available observation is used as the translation origin.
    No true fixed point or reference LE enters this transformation.
    """
    origin = data[-1]

    scale = max(
        mp.fabs(v - origin)
        for v in data[:-1]
    )

    if scale == 0:
        raise ValueError(
            "Observed record is constant relative to its last observation."
        )

    z = np.asarray(
        [
            float((v - origin) / scale)
            for v in data
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(z)):
        raise ValueError(
            "Nonfinite value after float64 normalization."
        )

    if np.unique(z).size < 3:
        raise ValueError(
            "Too few distinct observations after float64 normalization."
        )

    return z


def run_method(name, data, K):
    if name == "Proposed-WB":
        return proposed_le_estimator(
            data,
            K=K,
            end_index=K,
            limit_method="next",
            numeric_backend="float64",
        )

    if name == "Proposed-Aitken":
        return proposed_le_estimator(
            data,
            K=K,
            end_index=K,
            limit_method="aitken",
            numeric_backend="float64",
        )

    if name == "Yang-Wu":
        return yang_wu_scalar_le_estimator(
            data,
            evolution=1,
            r_min=None,
            numeric_backend="float64",
        )

    if name == "Rosenstein":
        return rosenstein_le_estimator(
            data,
            embedding_dim=1,
            lag=1,
            temporal_exclusion=0,
            sample_time=1.0,
            max_horizon=None,
            numeric_backend="float64",
        )

    if name == "Sun-Wu":
        return sun_wu_scalar_rbf_le_estimator(
            data,
            max_hidden_nodes=20,
            width_neighbors=None,
        )

    raise ValueError(f"Unknown estimator: {name}")


def warm_up():
    """
    One unreported warm-up call per estimator.
    """
    x = generate_trajectory(
        mp.mpf("7.0e9"),
        60,
    )

    K = 20
    n = 60

    data = normalize_observed_record_to_float64(
        x[n-K:n+2]
    )

    for method in METHODS:
        result = run_method(
            method,
            data,
            K,
        )

        if not result.get("valid", False):
            raise RuntimeError(
                f"Warm-up failed for {method}: "
                f"{result.get('failure_reason', '')}"
            )


def timed_method(name, data, K):
    """
    Time only the estimator call.
    """
    gc.collect()

    started = time.perf_counter()
    result = run_method(name, data, K)
    elapsed = time.perf_counter() - started

    value = result.get("lambda_hat", math.nan)

    try:
        value_float = float(value)
    except Exception:
        value_float = math.nan

    valid = bool(
        result.get("valid", False)
        and math.isfinite(value_float)
    )

    reason = result.get("failure_reason", "")

    if not valid and not reason:
        reason = (
            "Estimator returned an invalid or "
            "nonfinite result."
        )

    return (
        result,
        value_float,
        valid,
        reason,
        elapsed,
    )


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def median(values):
    values = list(values)
    return (
        statistics.median(values)
        if values
        else math.nan
    )


def mean(values):
    values = list(values)
    return (
        statistics.fmean(values)
        if values
        else math.nan
    )


def summarize(rows):
    summary = []

    for K in sorted({r["K"] for r in rows}):
        for method in METHODS:
            group = [
                r for r in rows
                if r["K"] == K
                and r["estimator"] == method
            ]

            valid = [
                r for r in group
                if r["valid"]
            ]

            times = [
                r["runtime_sec"]
                for r in group
            ]

            errors = [
                r["absolute_error"]
                for r in valid
            ]

            summary.append({
                "K": K,
                "paper_aligned_K": K in (10, 20),
                "estimator": method,
                "n_cases": len(group),
                "n_valid": len(valid),
                "valid_rate":
                    len(valid) / len(group)
                    if group else math.nan,
                "mean_absolute_error":
                    mean(errors),
                "median_absolute_error":
                    median(errors),
                "max_absolute_error":
                    max(errors)
                    if errors else math.nan,
                "median_runtime_sec":
                    median(times),
                "mean_runtime_sec":
                    mean(times),
                "min_runtime_sec":
                    min(times)
                    if times else math.nan,
                "max_runtime_sec":
                    max(times)
                    if times else math.nan,
            })

    # Overall runtime summary across every K and replicate.
    for method in METHODS:
        group = [
            r for r in rows
            if r["estimator"] == method
        ]

        valid = [
            r for r in group
            if r["valid"]
        ]

        times = [
            r["runtime_sec"]
            for r in group
        ]

        errors = [
            r["absolute_error"]
            for r in valid
        ]

        summary.append({
            "K": "ALL",
            "paper_aligned_K": False,
            "estimator": method,
            "n_cases": len(group),
            "n_valid": len(valid),
            "valid_rate":
                len(valid) / len(group)
                if group else math.nan,
            "mean_absolute_error":
                mean(errors),
            "median_absolute_error":
                median(errors),
            "max_absolute_error":
                max(errors)
                if errors else math.nan,
            "median_runtime_sec":
                median(times),
            "mean_runtime_sec":
                mean(times),
            "min_runtime_sec":
                min(times)
                if times else math.nan,
            "max_runtime_sec":
                max(times)
                if times else math.nan,
        })

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[10, 20, 40, 50],
    )

    parser.add_argument(
        "--replicates",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--dps",
        type=int,
        default=220,
        help=(
            "Precision used only for trajectory generation "
            "and pre-float normalization."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/experiment4_newton_control_runtime"
        ),
    )

    args = parser.parse_args()

    if args.steps < 2:
        parser.error(
            "steps must be >= 2"
        )

    if (
        not args.windows
        or min(args.windows) < 2
        or max(args.windows) > args.steps
    ):
        parser.error(
            "require 2 <= K <= steps"
        )

    if args.replicates < 1:
        parser.error(
            "replicates must be >= 1"
        )

    mp.mp.dps = args.dps

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Warm-up is deliberately outside all reported timings.
    warm_up()

    lambda_true = float(
        mp.log(mp.mpf("0.5"))
    )

    rows = []

    for replicate, x0 in enumerate(
        initial_points(
            args.seed,
            args.replicates,
        ),
        start=1,
    ):
        trajectory = generate_trajectory(
            x0,
            args.steps,
        )

        for K in args.windows:
            n = args.steps

            raw = trajectory[
                n-K:n+2
            ]

            # Normalization is not timed.
            data = (
                normalize_observed_record_to_float64(
                    raw
                )
            )

            for method in METHODS:
                try:
                    (
                        result,
                        value,
                        valid,
                        reason,
                        elapsed,
                    ) = timed_method(
                        method,
                        data,
                        K,
                    )

                except Exception as error:
                    result = {}
                    value = math.nan
                    valid = False
                    reason = (
                        f"{type(error).__name__}: "
                        f"{error}"
                    )
                    elapsed = 0.0

                rows.append({
                    "replicate": replicate,
                    "x0": float(x0),
                    "K": K,
                    "paper_aligned_K":
                        K in (10, 20),
                    "estimator": method,
                    "lambda_true":
                        lambda_true,
                    "lambda_hat":
                        value,
                    "absolute_error":
                        abs(value-lambda_true)
                        if valid
                        else math.nan,
                    "correct_sign":
                        bool(
                            valid
                            and value < 0
                        ),
                    "valid":
                        valid,
                    "failure_reason":
                        reason,
                    "runtime_sec":
                        elapsed,
                })

        print(
            f"Completed replicate="
            f"{replicate:02d}",
            flush=True,
        )

    summary = summarize(rows)

    write_csv(
        args.output
        / "runtime_detail.csv",
        rows,
    )

    write_csv(
        args.output
        / "runtime_summary.csv",
        summary,
    )

    config = {
        "experiment":
            "Newton control runtime benchmark",
        "function":
            "(x-2)^2",
        "newton_update":
            "(x+2)/2",
        "lambda_true":
            lambda_true,
        "steps":
            args.steps,
        "windows":
            args.windows,
        "replicates":
            args.replicates,
        "seed":
            args.seed,
        "trajectory_precision_dps":
            args.dps,
        "estimator_numeric_representation":
            "float64",
        "normalization":
            "(x_i - x_last) / "
            "max_j |x_j - x_last|, "
            "computed in mpmath before float64 conversion",
        "warmup":
            "one unreported call per estimator",
        "timed_region":
            "estimator call only",
        "excluded_from_timing": [
            "trajectory generation",
            "reference calculation",
            "normalization",
            "float64 conversion",
            "CSV writing",
        ],
        "primary_runtime_statistic":
            "median",
        "python":
            platform.python_version(),
        "numpy":
            np.__version__,
        "mpmath":
            mp.__version__,
    }

    (
        args.output
        / "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== RUNTIME SUMMARY ==="
    )

    for row in summary:
        if row["K"] == "ALL":
            continue

        print(
            f"K={row['K']:2} "
            f"{row['estimator']:<18} "
            f"valid={row['n_valid']}/"
            f"{row['n_cases']} "
            f"median="
            f"{row['median_runtime_sec']:.6f}s"
        )

    print(
        "\nOverall medians:"
    )

    for row in summary:
        if row["K"] != "ALL":
            continue

        print(
            f"{row['estimator']:<18} "
            f"{row['median_runtime_sec']:.6f}s"
        )


if __name__ == "__main__":
    main()
