"""
Experiment 3B: per-sample loss Lyapunov estimation and runtime benchmark.

Purpose
-------
Estimate a Lyapunov exponent separately from each training sample's scalar
squared-loss trajectory, and measure the total estimator cost of processing
all informative sample-loss trajectories in a small multi-parameter regression
problem.

This is a separate runtime/observation experiment from the batch-loss study.

For the controlled regression construction,

    w_t - w* = a^t (w_0 - w*)

and sample i has residual

    e_i(t) = a^t b_i - sigma h_i,

where b_i = x_i^T (w_0-w*) and h_i is the unused Hadamard column.

The individual sample loss is

    l_i(t) = 0.5 * e_i(t)^2,

with limit

    l_i(*) = sigma^2 / 2.

For sigma > 0 and b_i != 0,

    l_i(t) - l_i(*) =
        -sigma*h_i*b_i*a^t + 0.5*b_i^2*a^(2t),

so the asymptotic scalar sample-loss multiplier is a and

    lambda_sample = log(|a|).

For sigma = 0 and b_i != 0, the first-order term is absent and

    lambda_sample = 2*log(|a|).

If b_i = 0, that sample loss is constant at its limiting value and contains
no convergence signal; it is recorded as degenerate and is not sent to an LE
estimator.

Fair runtime representation
---------------------------
Each final K+2 scalar sample-loss record is translated and scaled in mpmath
before float64 conversion:

    z_j = (loss_j - loss_last) / max_q |loss_q - loss_last|.

This uses observed values only and gives every estimator the same float64
record. The transformation is affine and does not change a scalar Lyapunov
exponent.

Each estimator is warmed up once outside timing. Each informative sample-loss
trajectory is then timed once. The main workload metric is the SUM of all
sample-level estimator runtimes for one (seed, a, K) learning problem. Because
a learning problem contains many sample-loss trajectories, this total-workload
time is more stable and more relevant than tiny differences between individual
sub-millisecond calls.

Run from repository root:
    python -m experiments.experiment3_sample_loss_runtime
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
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


def hadamard(n: int):
    if n < 2 or n & (n - 1):
        raise ValueError("samples must be a power of two >= 2")
    H = [[1]]
    while len(H) < n:
        H = [r + r for r in H] + [r + [-v for v in r] for r in H]
    return H


def train_with_sample_losses(
    a,
    steps,
    seed,
    samples=16,
    parameters=8,
    sigma="0.3",
):
    """Run the same controlled regression training and retain per-sample losses."""
    if not 1 < parameters < samples:
        raise ValueError("require 1 < parameters < samples")

    a = mp.mpf(str(a))
    sigma = mp.mpf(str(sigma))

    if not 0 < abs(a) < 1:
        raise ValueError("require 0 < |a| < 1")
    if sigma < 0:
        raise ValueError("require sigma >= 0")

    rng = random.Random(seed)
    H = hadamard(samples)

    cols = list(range(samples))
    rng.shuffle(cols)

    signs = [
        rng.choice([-1, 1])
        for _ in range(parameters)
    ]

    X = mp.matrix([
        [
            signs[j] * H[i][cols[j]]
            for j in range(parameters)
        ]
        for i in range(samples)
    ])

    wstar = mp.matrix([
        mp.mpf(rng.randint(-20, 20)) / 10
        for _ in range(parameters)
    ])

    offset = mp.matrix([
        mp.mpf(rng.choice([-9, -5, 3, 7])) / 10
        for _ in range(parameters)
    ])

    w = wstar + offset

    h = [
        mp.mpf(H[i][cols[parameters]])
        for i in range(samples)
    ]

    noise = mp.matrix([
        sigma * h[i]
        for i in range(samples)
    ])

    y = X * wstar + noise
    learning_rate = 1 - a

    batch_losses = []
    sample_losses = [
        []
        for _ in range(samples)
    ]

    for t in range(steps + 2):
        residual = X * w - y

        row_losses = [
            residual[i] ** 2 / 2
            for i in range(samples)
        ]

        for i, value in enumerate(row_losses):
            sample_losses[i].append(value)

        batch_losses.append(
            mp.fsum(row_losses) / samples
        )

        if t < steps + 1:
            gradient = X.T * residual / samples
            w = w - learning_rate * gradient

    # b_i controls whether sample i contains a convergence signal.
    b = X * offset

    loss_limit = sigma ** 2 / 2

    if sigma > 0:
        sample_le_true = mp.log(abs(a))
    else:
        sample_le_true = 2 * mp.log(abs(a))

    batch_le_true = 2 * mp.log(abs(a))

    diagnostics = []
    for i in range(samples):
        informative = (b[i] != 0)

        diagnostics.append({
            "seed": seed,
            "a": a,
            "sample_index": i,
            "b_i": b[i],
            "hadamard_noise_sign": h[i],
            "informative": informative,
            "sample_loss_limit_true": loss_limit,
            "sample_loss_le_true":
                sample_le_true if informative else mp.nan,
        })

    return {
        "batch_losses": batch_losses,
        "sample_losses": sample_losses,
        "diagnostics": diagnostics,
        "sample_loss_limit_true": loss_limit,
        "sample_loss_le_true": sample_le_true,
        "batch_loss_le_true": batch_le_true,
        "learning_rate": learning_rate,
    }


def normalize_observed_record_to_float64(data):
    """Observed-only high-precision affine normalization."""
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
        dtype=float,
    )

    if not np.all(np.isfinite(z)):
        raise ValueError(
            "Nonfinite value after float64 normalization."
        )

    # Exact duplicate observations are allowed in principle, but for this
    # controlled final-window benchmark they indicate a loss of usable geometry.
    if np.unique(z).size < 3:
        raise ValueError(
            "Too few distinct float64 observations after normalization."
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

    raise ValueError(f"Unknown estimator {name}")


def timed_method(name, data, K):
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

    return result, value_float, valid, elapsed


def warm_up():
    """Unreported warm-up call for every estimator."""
    problem = train_with_sample_losses(
        a="0.95",
        steps=60,
        seed=2027,
        sigma="0.3",
    )

    # Find the first informative sample.
    sample_index = next(
        d["sample_index"]
        for d in problem["diagnostics"]
        if d["informative"]
    )

    K = 20
    n = 60

    raw = problem["sample_losses"][sample_index][n-K:n+2]
    data = normalize_observed_record_to_float64(raw)

    for method in METHODS:
        result = run_method(method, data, K)
        if not result.get("valid", False):
            raise RuntimeError(
                f"Warm-up failed for {method}: "
                f"{result.get('failure_reason', '')}"
            )


def text_value(value):
    if isinstance(value, mp.mpf):
        return mp.nstr(value, 55)
    return value


def write_csv(path, rows):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        writer.writerows([
            {
                k: text_value(v)
                for k, v in row.items()
            }
            for row in rows
        ])


def median(values):
    values = list(values)
    return statistics.median(values) if values else math.nan


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def summarize_calls(rows):
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
                "estimator": method,
                "n_calls": len(group),
                "n_valid": len(valid),
                "valid_rate":
                    len(valid) / len(group)
                    if group else math.nan,
                "mean_absolute_error":
                    mean(errors),
                "median_absolute_error":
                    median(errors),
                "max_absolute_error":
                    max(errors) if errors else math.nan,
                "median_runtime_per_sample_sec":
                    median(times),
                "mean_runtime_per_sample_sec":
                    mean(times),
            })

    return summary


def summarize_workloads(rows):
    """
    One workload = estimate LE for every informative sample in one
    (seed, a, K) training problem.
    """
    workloads = []

    keys = sorted({
        (
            r["seed"],
            r["a"],
            r["K"],
            r["estimator"],
        )
        for r in rows
    })

    for seed, a, K, method in keys:
        group = [
            r for r in rows
            if r["seed"] == seed
            and r["a"] == a
            and r["K"] == K
            and r["estimator"] == method
        ]

        workloads.append({
            "seed": seed,
            "a": a,
            "K": K,
            "estimator": method,
            "n_sample_losses": len(group),
            "n_valid": sum(
                bool(r["valid"])
                for r in group
            ),
            "total_runtime_sec":
                sum(r["runtime_sec"] for r in group),
            "mean_runtime_per_sample_sec":
                mean(r["runtime_sec"] for r in group),
            "median_runtime_per_sample_sec":
                median(r["runtime_sec"] for r in group),
        })

    aggregate = []

    for K in sorted({r["K"] for r in workloads}):
        for method in METHODS:
            group = [
                r for r in workloads
                if r["K"] == K
                and r["estimator"] == method
            ]

            totals = [
                r["total_runtime_sec"]
                for r in group
            ]

            aggregate.append({
                "K": K,
                "estimator": method,
                "n_learning_problems": len(group),
                "median_informative_samples":
                    median(r["n_sample_losses"] for r in group),
                "median_total_runtime_sec":
                    median(totals),
                "mean_total_runtime_sec":
                    mean(totals),
                "min_total_runtime_sec":
                    min(totals) if totals else math.nan,
                "max_total_runtime_sec":
                    max(totals) if totals else math.nan,
            })

    # Overall across K.
    for method in METHODS:
        group = [
            r for r in workloads
            if r["estimator"] == method
        ]

        totals = [
            r["total_runtime_sec"]
            for r in group
        ]

        aggregate.append({
            "K": "ALL",
            "estimator": method,
            "n_learning_problems": len(group),
            "median_informative_samples":
                median(r["n_sample_losses"] for r in group),
            "median_total_runtime_sec":
                median(totals),
            "mean_total_runtime_sec":
                mean(totals),
            "min_total_runtime_sec":
                min(totals) if totals else math.nan,
            "max_total_runtime_sec":
                max(totals) if totals else math.nan,
        })

    return workloads, aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--a-values",
        nargs="+",
        default=[
            "-0.95",
            "-0.6",
            "-0.2",
            "0.2",
            "0.6",
            "0.95",
        ],
    )

    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[10, 20, 40, 50],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[2026, 2027],
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--parameters",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--sigma",
        default="0.3",
    )

    parser.add_argument(
        "--dps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/experiment3_sample_loss_runtime"
        ),
    )

    args = parser.parse_args()

    if max(args.windows) > args.steps:
        parser.error("require max(K) <= steps")

    # Keep enough digits for both batch a^(2t) behavior and sample a^t behavior.
    needed = int(max(
        -2 * (args.steps + 1)
        * mp.log10(abs(mp.mpf(a)))
        for a in args.a_values
    )) + 80

    mp.mp.dps = (
        args.dps
        or max(120, needed)
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    warm_up()

    detail = []
    sample_diagnostics = []

    for seed in args.seeds:
        for a_text in args.a_values:
            problem = train_with_sample_losses(
                a=a_text,
                steps=args.steps,
                seed=seed,
                samples=args.samples,
                parameters=args.parameters,
                sigma=args.sigma,
            )

            truth = problem["sample_loss_le_true"]

            for d in problem["diagnostics"]:
                sample_diagnostics.append(d)

            informative_indices = [
                d["sample_index"]
                for d in problem["diagnostics"]
                if d["informative"]
            ]

            print(
                f"seed={seed}, a={a_text}: "
                f"{len(informative_indices)}/"
                f"{args.samples} informative sample losses",
                flush=True,
            )

            for K in args.windows:
                n = args.steps

                for sample_index in informative_indices:
                    raw = problem["sample_losses"][
                        sample_index
                    ][n-K:n+2]

                    try:
                        data = (
                            normalize_observed_record_to_float64(
                                raw
                            )
                        )
                    except Exception as error:
                        for method in METHODS:
                            detail.append({
                                "seed": seed,
                                "a": float(mp.mpf(a_text)),
                                "K": K,
                                "sample_index": sample_index,
                                "estimator": method,
                                "sample_loss_le_true":
                                    float(truth),
                                "lambda_hat": math.nan,
                                "absolute_error": math.nan,
                                "valid": False,
                                "failure_reason":
                                    f"normalization: "
                                    f"{type(error).__name__}: "
                                    f"{error}",
                                "runtime_sec": 0.0,
                            })
                        continue

                    for method in METHODS:
                        try:
                            (
                                result,
                                value,
                                valid,
                                elapsed,
                            ) = timed_method(
                                method,
                                data,
                                K,
                            )

                            reason = result.get(
                                "failure_reason",
                                "",
                            )

                        except Exception as error:
                            value = math.nan
                            valid = False
                            elapsed = 0.0
                            reason = (
                                f"{type(error).__name__}: "
                                f"{error}"
                            )

                        error_value = (
                            abs(value - float(truth))
                            if valid
                            else math.nan
                        )

                        detail.append({
                            "seed": seed,
                            "a": float(mp.mpf(a_text)),
                            "K": K,
                            "sample_index": sample_index,
                            "estimator": method,
                            "sample_loss_le_true":
                                float(truth),
                            "lambda_hat": value,
                            "absolute_error": error_value,
                            "valid": valid,
                            "failure_reason": reason,
                            "runtime_sec": elapsed,
                        })

    call_summary = summarize_calls(detail)
    workloads, workload_summary = summarize_workloads(
        detail
    )

    write_csv(
        args.output / "sample_loss_runtime_detail.csv",
        detail,
    )

    write_csv(
        args.output / "sample_loss_diagnostics.csv",
        sample_diagnostics,
    )

    write_csv(
        args.output / "sample_loss_call_summary.csv",
        call_summary,
    )

    write_csv(
        args.output / "sample_loss_workloads.csv",
        workloads,
    )

    write_csv(
        args.output / "sample_loss_workload_summary.csv",
        workload_summary,
    )

    print(
        "\n=== MEDIAN TOTAL RUNTIME PER LEARNING PROBLEM ==="
    )
    print(
        "(sum over all informative sample-loss trajectories)"
    )

    overall = [
        r
        for r in workload_summary
        if r["K"] == "ALL"
    ]

    for r in overall:
        print(
            f"{r['estimator']:<18} "
            f"median_total="
            f"{r['median_total_runtime_sec']:.6f}s "
            f"mean_total="
            f"{r['mean_total_runtime_sec']:.6f}s"
        )

    print("\nSaved to:")
    print(args.output)


if __name__ == "__main__":
    main()
