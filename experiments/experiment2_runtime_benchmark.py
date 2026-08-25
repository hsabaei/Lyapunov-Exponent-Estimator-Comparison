"""
Experiment 2: fair float64 runtime benchmark for nonlinear contraction.

This timing experiment is separate from the high-precision accuracy benchmark.

System
------
    x_{t+1} - L = a (x_t - L) + beta (x_t - L)^2

with the same a, beta, eta, K, N, L, and x0 used in the nonlinear
accuracy comparison.

Fair timing representation
--------------------------
The mathematical accuracy benchmark uses different arithmetic backends across
methods, so those runtimes are diagnostic only.

For fair timing, every estimator receives the SAME float64 representation of
the same final K+2 observations. Before converting to float64, the window is
translated and scaled in high precision:

    z_i = (x_i - x_last) / s,
    s   = max_i |x_i - x_last|.

This uses only the observed window. It does NOT use the true fixed point L or
the true Lyapunov exponent.

Anchoring at the last observation preserves the small contracting differences
much better than anchoring at the first observation.

Each estimator is warmed up once outside the recorded timing. Each
(a, beta, K, estimator) configuration is then timed once.

Primary runtime summary:
    - median runtime overall
    - mean/min/max runtime overall
    - median runtime by eta
    - median runtime by K

Because individual configurations are timed once, emphasize broad runtime
differences rather than tiny sub-millisecond differences between closely
related methods.
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

METHODS = [
    "Proposed-Next",
    "Proposed-Aitken",
    "Yang-Wu",
    "Sun-Wu",
    "Rosenstein",
]

OUTPUT_DIR = Path("results/tables")
DETAIL_PATH = OUTPUT_DIR / "experiment2_runtime_benchmark.csv"
SUMMARY_PATH = OUTPUT_DIR / "experiment2_runtime_summary.csv"


def nonlinear_beta_cases(a):
    a = mp.mpf(a)
    d = mp.fabs(X0 - L)

    out = [
        (
            mp.mpf("0.0"),
            0,
            mp.mpf("0.0"),
            mp.fabs(a),
        )
    ]

    for eta_text in ["0.4", "0.9"]:
        eta = mp.mpf(eta_text)

        beta_abs = (
            eta
            * (mp.mpf("1.0") - mp.fabs(a))
            / (mp.mpf("2.0") * d)
        )

        q_bound = (
            mp.fabs(a)
            + eta * (mp.mpf("1.0") - mp.fabs(a))
        )

        if not q_bound < 1:
            raise RuntimeError(
                f"Expected contraction bound q<1, got {q_bound}."
            )

        for beta_sign in (-1, +1):
            out.append(
                (
                    eta,
                    beta_sign,
                    mp.mpf(beta_sign) * beta_abs,
                    q_bound,
                )
            )

    return out


def generate_nonlinear_trajectory(a, beta):
    a = mp.mpf(a)
    beta = mp.mpf(beta)

    x = [X0]

    for _ in range(N + 1):
        e = x[-1] - L
        x_next = L + a * e + beta * e * e
        x.append(x_next)

    return x


def final_window(x, K):
    # x_{N-K}, ..., x_N, x_{N+1}
    return x[N - K : N + 2]


def normalize_window_to_float64(data):
    """
    Observed-data-only affine coordinate transform:

        z_i = (x_i - x_last) / scale

    where scale is the maximum absolute difference from x_last.

    The transform is performed in mpmath BEFORE float64 conversion.
    """
    origin = data[-1]
    scale = max(
        mp.fabs(v - origin)
        for v in data[:-1]
    )

    if scale == 0:
        raise ValueError(
            "Observed window has zero scale."
        )

    z = [
        (v - origin) / scale
        for v in data
    ]

    out = np.asarray(
        [float(v) for v in z],
        dtype=float,
    )

    if not np.all(np.isfinite(out)):
        raise ValueError(
            "Nonfinite value after float64 normalization."
        )

    if np.unique(out).size != out.size:
        raise ValueError(
            "Last-anchored normalization did not preserve "
            "distinct observations in float64."
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

    raise ValueError(
        f"Unknown method: {name}"
    )


def timed_call(name, data, K):
    gc.collect()

    start = time.perf_counter()
    result = run_method(
        name=name,
        data=data,
        K=K,
    )
    elapsed = time.perf_counter() - start

    return result, elapsed


def warm_up():
    """
    One unreported call per method.
    Use a representative nonlinear case.
    """
    a = mp.mpf("0.8")
    eta = mp.mpf("0.9")
    d = mp.fabs(X0 - L)

    beta = (
        eta
        * (mp.mpf("1.0") - mp.fabs(a))
        / (mp.mpf("2.0") * d)
    )

    K = 20

    x = generate_nonlinear_trajectory(
        a=a,
        beta=beta,
    )

    data = normalize_window_to_float64(
        final_window(x, K)
    )

    print("Warming up estimators...")

    for name in METHODS:
        result = run_method(
            name=name,
            data=data,
            K=K,
        )

        if not result.get("valid", False):
            raise RuntimeError(
                f"Warm-up failed for {name}: "
                f"{result.get('failure_reason')}"
            )


def write_csv(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def summarize_group(
    *,
    scope,
    eta_label,
    K_label,
    estimator,
    subset,
):
    times = [
        r["runtime_sec"]
        for r in subset
    ]

    return {
        "scope": scope,
        "eta": eta_label,
        "K": K_label,
        "estimator": estimator,
        "n_cases": len(subset),
        "n_valid": sum(
            bool(r["valid"])
            for r in subset
        ),
        "median_runtime_sec":
            statistics.median(times),
        "mean_runtime_sec":
            statistics.fmean(times),
        "min_runtime_sec":
            min(times),
        "max_runtime_sec":
            max(times),
        "total_runtime_sec":
            sum(times),
    }


def make_summary(rows):
    summary = []

    # Overall.
    for method in METHODS:
        subset = [
            r
            for r in rows
            if r["estimator"] == method
        ]

        summary.append(
            summarize_group(
                scope="overall",
                eta_label="ALL",
                K_label="ALL",
                estimator=method,
                subset=subset,
            )
        )

    # By eta.
    for eta in [0.0, 0.4, 0.9]:
        for method in METHODS:
            subset = [
                r
                for r in rows
                if r["estimator"] == method
                and r["eta"] == eta
            ]

            summary.append(
                summarize_group(
                    scope="by_eta",
                    eta_label=eta,
                    K_label="ALL",
                    estimator=method,
                    subset=subset,
                )
            )

    # By K.
    for K in K_VALUES:
        for method in METHODS:
            subset = [
                r
                for r in rows
                if r["estimator"] == method
                and r["K"] == K
            ]

            summary.append(
                summarize_group(
                    scope="by_K",
                    eta_label="ALL",
                    K_label=K,
                    estimator=method,
                    subset=subset,
                )
            )

    return summary


def main():
    mp.dps = MP_DPS

    warm_up()

    rows = []

    total = (
        len(A_VALUES)
        * 5
        * len(K_VALUES)
        * len(METHODS)
    )

    completed = 0

    for a_text in A_VALUES:
        a = mp.mpf(a_text)
        truth = mp.log(mp.fabs(a))

        for (
            eta,
            beta_sign,
            beta,
            q_bound,
        ) in nonlinear_beta_cases(a):

            x = generate_nonlinear_trajectory(
                a=a,
                beta=beta,
            )

            for K in K_VALUES:
                original = final_window(
                    x=x,
                    K=K,
                )

                data = normalize_window_to_float64(
                    original
                )

                for method in METHODS:
                    result, elapsed = timed_call(
                        name=method,
                        data=data,
                        K=K,
                    )

                    estimate = result.get(
                        "lambda_hat",
                        math.nan,
                    )

                    try:
                        estimate_float = float(
                            estimate
                        )
                    except Exception:
                        estimate_float = math.nan

                    if (
                        result.get("valid", False)
                        and math.isfinite(
                            estimate_float
                        )
                    ):
                        abs_error = abs(
                            estimate_float
                            - float(truth)
                        )
                    else:
                        abs_error = math.nan

                    rows.append(
                        {
                            "a": float(a),
                            "eta": float(eta),
                            "beta_sign": beta_sign,
                            "beta": float(beta),
                            "q_bound": float(q_bound),
                            "K": K,
                            "estimator": method,
                            "runtime_sec": elapsed,
                            "valid": bool(
                                result.get(
                                    "valid",
                                    False,
                                )
                            ),
                            "failure_reason":
                                result.get(
                                    "failure_reason"
                                ),
                            "lambda_true":
                                float(truth),
                            "lambda_hat":
                                estimate_float,
                            "abs_error_after_normalization":
                                abs_error,
                        }
                    )

                    completed += 1

                    print(
                        f"completed "
                        f"{completed}/{total}: "
                        f"a={float(a):+.2f}, "
                        f"eta={float(eta):.1f}, "
                        f"beta_sign={beta_sign:+d}, "
                        f"K={K}, "
                        f"{method}, "
                        f"time={elapsed:.6f}s"
                    )

    summary = make_summary(rows)

    write_csv(
        DETAIL_PATH,
        rows,
    )

    write_csv(
        SUMMARY_PATH,
        summary,
    )

    print(
        "\n=== EXPERIMENT 2 FAIR FLOAT64 "
        "RUNTIME SUMMARY ==="
    )

    print("\nOverall:")

    for r in summary:
        if r["scope"] != "overall":
            continue

        print(
            f"{r['estimator']:<18} "
            f"median="
            f"{r['median_runtime_sec']:.6f}s "
            f"mean="
            f"{r['mean_runtime_sec']:.6f}s "
            f"min="
            f"{r['min_runtime_sec']:.6f}s "
            f"max="
            f"{r['max_runtime_sec']:.6f}s "
            f"valid="
            f"{r['n_valid']}/"
            f"{r['n_cases']}"
        )

    print("\nMedian runtime by eta:")

    for eta in [0.0, 0.4, 0.9]:
        print(f"\neta={eta:.1f}")

        for r in summary:
            if (
                r["scope"] == "by_eta"
                and r["eta"] == eta
            ):
                print(
                    f"  {r['estimator']:<18} "
                    f"{r['median_runtime_sec']:.6f}s"
                )

    print("\nMedian runtime by K:")

    for K in K_VALUES:
        print(f"\nK={K}")

        for r in summary:
            if (
                r["scope"] == "by_K"
                and r["K"] == K
            ):
                print(
                    f"  {r['estimator']:<18} "
                    f"{r['median_runtime_sec']:.6f}s"
                )

    print("\nSaved:")
    print(DETAIL_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
