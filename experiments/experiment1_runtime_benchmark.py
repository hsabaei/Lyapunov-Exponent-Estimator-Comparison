"""
Fair runtime benchmark for Experiment 1.

Why this is separate from the high-precision accuracy benchmark
---------------------------------------------------------------
The mathematical validation uses different arithmetic backends:
- Proposed, Yang-Wu, Rosenstein: mpmath
- Sun-Wu: normalized float64 RBF fitting

Those wall-clock times are useful diagnostics but are not an apples-to-apples
speed comparison.

For timing, every method is therefore run on the SAME float64 representation
of the same final K+2 observations. Before converting to float64, each window
is translated and scaled using only the observed window itself:

    z_i = (x_i - x_last) / s,
    s   = max_i |x_i - x_last|.

This prevents stable fixed-point tails from collapsing to identical float64
values. Translation and nonzero scalar rescaling do not change the Lyapunov
exponent, and the transform uses no true fixed point or true LE.

Each estimator is warmed up once outside the reported timings. Then each of
the 40 (a, K) configurations is timed once. Overall medians aggregate over 40
independent configuration timings; per-K medians aggregate over the 10 values
of a. This keeps the timing experiment practical, especially because the
Sun-Wu architecture search is comparatively expensive.
"""

from __future__ import annotations

import csv
import gc
import math
import statistics
import time
from pathlib import Path

import numpy as np
from mpmath import mp

from src.estimators.proposed import proposed_le_estimator
from src.estimators.yang_wu import yang_wu_scalar_le_estimator
from src.estimators.sun_wu import sun_wu_scalar_rbf_le_estimator
from src.estimators.rosenstein import rosenstein_le_estimator


MP_DPS = 200
N = 200

L = mp.mpf("2.0")
X0 = mp.mpf("1.3578696273508724")

A_VALUES = [
    "-0.95",
    "-0.8",
    "-0.6",
    "-0.4",
    "-0.2",
    "0.2",
    "0.4",
    "0.6",
    "0.8",
    "0.95",
]

K_VALUES = [10, 20, 40, 50]

OUTPUT_DIR = Path("results/tables")
DETAIL_PATH = OUTPUT_DIR / "experiment1_runtime_benchmark.csv"
SUMMARY_PATH = OUTPUT_DIR / "experiment1_runtime_summary.csv"


def affine_trajectory(a):
    a = mp.mpf(a)
    x = [X0]

    # x_0, ..., x_N, x_{N+1}
    for _ in range(N + 1):
        x.append(L + a * (x[-1] - L))

    return x


def final_window(x, K):
    # x_{N-K}, ..., x_N, x_{N+1}
    return x[N - K : N + 2]


def normalize_window_to_float64(data):
    """
    Observed-data-only affine coordinate transform.

    Anchor at the LAST observed value, not the first:

        z_i = (x_i - x_last) / s,
        s   = max_i |x_i - x_last|.

    Why this matters:
    For a strongly contracting tail, anchoring at the first point makes many
    later normalized values round to 1.0 in float64. Anchoring at the last
    point keeps the shrinking differences near 0, where float64 can still
    represent them distinctly over the present benchmark range.

    The subtraction and scaling are done in mpmath BEFORE conversion to
    float64. The transform uses only observed data; it does not use the true
    fixed point or the true Lyapunov exponent.
    """
    origin = data[-1]
    scale = max(mp.fabs(v - origin) for v in data[:-1])

    if scale == 0:
        raise ValueError("Observed window has zero scale.")

    z = [(v - origin) / scale for v in data]
    out = np.asarray([float(v) for v in z], dtype=float)

    if not np.all(np.isfinite(out)):
        raise ValueError("Nonfinite value after float64 normalization.")

    if np.unique(out).size != out.size:
        raise ValueError(
            "Last-anchored normalization did not preserve distinct "
            "observations in float64."
        )

    return out


def run_method(name, data, K):
    if name == "Proposed-Next":
        return proposed_le_estimator(
            x=data,
            K=K,
            end_index=K,
            limit_method="next",
            numeric_backend="float64",
        )

    if name == "Proposed-Aitken":
        return proposed_le_estimator(
            x=data,
            K=K,
            end_index=K,
            limit_method="aitken",
            numeric_backend="float64",
        )

    if name == "Yang-Wu":
        return yang_wu_scalar_le_estimator(
            x=data,
            evolution=1,
            r_min=None,
            numeric_backend="float64",
        )

    if name == "Sun-Wu":
        return sun_wu_scalar_rbf_le_estimator(
            x=data,
            max_hidden_nodes=20,
            width_neighbors=None,
        )

    if name == "Rosenstein":
        return rosenstein_le_estimator(
            x=data,
            embedding_dim=1,
            lag=1,
            temporal_exclusion=0,
            sample_time=1.0,
            max_horizon=None,
            numeric_backend="float64",
        )

    raise ValueError(f"Unknown method: {name}")


METHODS = [
    "Proposed-Next",
    "Proposed-Aitken",
    "Yang-Wu",
    "Sun-Wu",
    "Rosenstein",
]


def timed_call(name, data, K):
    # Keep garbage collection outside the timed interval.
    gc.collect()

    start = time.perf_counter()
    result = run_method(name=name, data=data, K=K)
    elapsed = time.perf_counter() - start

    return result, elapsed


def warm_up():
    """
    One unreported call per estimator to reduce one-time startup effects.
    """
    a = mp.mpf("0.8")
    K = 20
    x = affine_trajectory(a)
    data = normalize_window_to_float64(final_window(x, K))

    print("Warming up estimators...")

    for name in METHODS:
        result = run_method(name=name, data=data, K=K)
        if not result.get("valid", False):
            raise RuntimeError(
                f"Warm-up failed for {name}: "
                f"{result.get('failure_reason')}"
            )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def make_summary(rows):
    summary = []

    # Overall rows.
    for method in METHODS:
        subset = [
            r for r in rows
            if r["estimator"] == method
        ]
        times = [r["runtime_sec"] for r in subset]

        summary.append(
            {
                "scope": "overall",
                "K": "ALL",
                "estimator": method,
                "n_cases": len(subset),
                "n_valid": sum(bool(r["valid"]) for r in subset),
                "median_runtime_sec": statistics.median(times),
                "mean_runtime_sec": statistics.fmean(times),
                "min_runtime_sec": min(times),
                "max_runtime_sec": max(times),
                "total_runtime_sec": sum(times),
            }
        )

    # Per-K rows.
    for K in K_VALUES:
        for method in METHODS:
            subset = [
                r for r in rows
                if r["estimator"] == method
                and r["K"] == K
            ]
            times = [r["runtime_sec"] for r in subset]

            summary.append(
                {
                    "scope": "by_K",
                    "K": K,
                    "estimator": method,
                    "n_cases": len(subset),
                    "n_valid": sum(bool(r["valid"]) for r in subset),
                    "median_runtime_sec": statistics.median(times),
                    "mean_runtime_sec": statistics.fmean(times),
                    "min_runtime_sec": min(times),
                    "max_runtime_sec": max(times),
                    "total_runtime_sec": sum(times),
                }
            )

    return summary


def main():
    mp.dps = MP_DPS

    warm_up()

    rows = []
    total = len(A_VALUES) * len(K_VALUES) * len(METHODS)
    completed = 0

    for a_text in A_VALUES:
        a = mp.mpf(a_text)
        truth = mp.log(mp.fabs(a))
        x = affine_trajectory(a)

        for K in K_VALUES:
            original = final_window(x, K)
            data = normalize_window_to_float64(original)

            for method in METHODS:
                result, elapsed = timed_call(
                    name=method,
                    data=data,
                    K=K,
                )

                estimate = result.get("lambda_hat", math.nan)

                try:
                    estimate_float = float(estimate)
                except Exception:
                    estimate_float = math.nan

                if (
                    result.get("valid", False)
                    and math.isfinite(estimate_float)
                ):
                    abs_error = abs(
                        estimate_float - float(truth)
                    )
                else:
                    abs_error = math.nan

                rows.append(
                    {
                        "a": float(a),
                        "K": K,
                        "estimator": method,
                        "runtime_sec": elapsed,
                        "valid": bool(result.get("valid", False)),
                        "failure_reason": result.get("failure_reason"),
                        "lambda_true": float(truth),
                        "lambda_hat": estimate_float,
                        "abs_error_after_normalization": abs_error,
                    }
                )

                completed += 1
                print(
                    f"completed {completed}/{total}: "
                    f"a={float(a):+.2f}, K={K}, "
                    f"{method}, time={elapsed:.6f}s"
                )

    summary = make_summary(rows)

    write_csv(DETAIL_PATH, rows)
    write_csv(SUMMARY_PATH, summary)

    print("\n=== FAIR FLOAT64 RUNTIME SUMMARY ===")

    overall = [
        r for r in summary
        if r["scope"] == "overall"
    ]

    for r in overall:
        print(
            f"{r['estimator']:<18} "
            f"median={r['median_runtime_sec']:.6f}s "
            f"mean={r['mean_runtime_sec']:.6f}s "
            f"min={r['min_runtime_sec']:.6f}s "
            f"max={r['max_runtime_sec']:.6f}s "
            f"valid={r['n_valid']}/{r['n_cases']}"
        )

    print("\n=== MEDIAN RUNTIME BY K ===")

    for K in K_VALUES:
        print(f"\nK={K}")
        subset = [
            r for r in summary
            if r["scope"] == "by_K"
            and r["K"] == K
        ]

        for r in subset:
            print(
                f"  {r['estimator']:<18} "
                f"{r['median_runtime_sec']:.6f}s"
            )

    print("\nSaved:")
    print(DETAIL_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
