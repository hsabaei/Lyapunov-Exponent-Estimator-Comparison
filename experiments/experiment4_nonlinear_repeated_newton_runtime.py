"""
Experiment 4B runtime benchmark: nonlinear repeated-root Newton family.

Family:
    f_m(x) = (x - 2)^m (x^2 + 1),
    m in {2,3,5}.

Asymptotic reference:
    U_m'(2) = (m-1)/m
    lambda_inf(m) = log((m-1)/m)

This script measures estimator runtime only.  It uses the same experiment
selection protocol as the high-precision accuracy experiment:

- common 10 initial points around 4.0e9 +/- 4.0e7,
- paper-style cutoff |x_t-2| < 1e-6,
- K in {10,20,40,50},
- exactly the same most-recent K+2 retained observations.

Fair runtime protocol
---------------------
Before float64 conversion, each observed K+2 record is normalized in mpmath:

    z_i = (x_i - x_last) / max_j |x_j - x_last|.

This transformation uses observed values only; it does not use the true fixed
point or the true LE.  Every estimator receives the exact same float64 record.

One unreported warm-up call per estimator is performed before timing.
The timed region includes only the estimator call.  Trajectory generation,
cutoff selection, normalization, float64 conversion, truth calculation and CSV
writing are excluded.

Run from repository root:

    python -m experiments.experiment4_nonlinear_repeated_newton_runtime
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


L_TRUE = mp.mpf("2")
DEFAULT_MULTIPLICITIES = [2, 3, 5]
METHODS = [
    "Proposed-WB",
    "Proposed-Aitken",
    "Yang-Wu",
    "Rosenstein",
    "Sun-Wu",
]


def newton_error_update(e, m):
    e = mp.mpf(e)
    m = int(m)

    if e == 0:
        return mp.mpf("0")

    numerator = (
        (m + 1) * e**2
        + 4 * m * e
        + 5 * (m - 1)
    )

    denominator = (
        (m + 2) * e**2
        + 4 * (m + 1) * e
        + 5 * m
    )

    if denominator == 0:
        raise ZeroDivisionError(
            "Newton error-map denominator is zero."
        )

    return e * numerator / denominator


def newton_update(x, m):
    return L_TRUE + newton_error_update(
        mp.mpf(x) - L_TRUE,
        m,
    )


def lambda_inf_true(m):
    return mp.log(
        mp.mpf(m - 1) / m
    )


def generate_until_cutoff(
    x0,
    m,
    cutoff,
    max_steps,
):
    x = [mp.mpf(x0)]

    for _ in range(max_steps):
        nxt = newton_update(
            x[-1],
            m,
        )
        x.append(nxt)

        if mp.fabs(
            nxt - L_TRUE
        ) < cutoff:
            return x

    raise RuntimeError(
        f"m={m}: cutoff not reached "
        f"within {max_steps} iterations."
    )


def common_initial_points(
    seed,
    replicates,
):
    rng = random.Random(seed)

    center = mp.mpf("4.0e9")
    half_width = mp.mpf("4.0e7")

    return [
        center
        + half_width
        * mp.mpf(
            str(
                rng.uniform(
                    -1.0,
                    1.0,
                )
            )
        )
        for _ in range(replicates)
    ]


def normalize_observed_record_to_float64(
    data,
):
    """
    Observed-only affine normalization.

    The last observed point is the translation origin.
    No true L or true lambda is used.
    """
    origin = data[-1]

    scale = max(
        mp.fabs(
            value - origin
        )
        for value in data[:-1]
    )

    if scale == 0:
        raise ValueError(
            "Observed record has zero scale."
        )

    z = np.asarray(
        [
            float(
                (value - origin)
                / scale
            )
            for value in data
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(z)
    ):
        raise ValueError(
            "Nonfinite normalized float64 record."
        )

    if np.unique(z).size < 3:
        raise ValueError(
            "Too few distinct float64 observations."
        )

    return z


def run_method(
    method,
    data,
    K,
):
    if method == "Proposed-WB":
        return proposed_le_estimator(
            data,
            K=K,
            end_index=K,
            limit_method="next",
            numeric_backend="float64",
        )

    if method == "Proposed-Aitken":
        return proposed_le_estimator(
            data,
            K=K,
            end_index=K,
            limit_method="aitken",
            numeric_backend="float64",
        )

    if method == "Yang-Wu":
        return yang_wu_scalar_le_estimator(
            data,
            evolution=1,
            r_min=None,
            numeric_backend="float64",
        )

    if method == "Rosenstein":
        return rosenstein_le_estimator(
            data,
            embedding_dim=1,
            lag=1,
            temporal_exclusion=0,
            sample_time=1.0,
            max_horizon=None,
            numeric_backend="float64",
        )

    if method == "Sun-Wu":
        return sun_wu_scalar_rbf_le_estimator(
            data,
            max_hidden_nodes=20,
            width_neighbors=None,
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


def warm_up():
    """
    One unreported call per estimator.
    """
    m = 3
    K = 20
    cutoff = mp.mpf("1e-6")

    x = generate_until_cutoff(
        x0=mp.mpf("4.0e9"),
        m=m,
        cutoff=cutoff,
        max_steps=5000,
    )

    last_retained = len(x) - 2
    latest = last_retained

    data_mp = x[
        latest - (K + 1):
        latest + 1
    ]

    if len(data_mp) != K + 2:
        raise RuntimeError(
            "Warm-up record has wrong length."
        )

    data = (
        normalize_observed_record_to_float64(
            data_mp
        )
    )

    for method in METHODS:
        result = run_method(
            method,
            data,
            K,
        )

        if not result.get(
            "valid",
            False,
        ):
            raise RuntimeError(
                f"Warm-up failed for {method}: "
                f"{result.get('failure_reason', '')}"
            )


def timed_method(
    method,
    data,
    K,
):
    gc.collect()

    start = time.perf_counter()

    result = run_method(
        method,
        data,
        K,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    value = result.get(
        "lambda_hat",
        math.nan,
    )

    try:
        value = float(value)
    except Exception:
        value = math.nan

    valid = bool(
        result.get(
            "valid",
            False,
        )
        and math.isfinite(value)
    )

    reason = result.get(
        "failure_reason",
        "",
    )

    if not valid and not reason:
        reason = (
            "Estimator returned invalid "
            "or nonfinite result."
        )

    return (
        value,
        valid,
        reason,
        elapsed,
    )


def write_csv(
    path,
    rows,
):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = list(values)
    return (
        statistics.fmean(values)
        if values
        else math.nan
    )


def median(values):
    values = list(values)
    return (
        statistics.median(values)
        if values
        else math.nan
    )


def summarize(rows):
    summary = []

    keys = sorted({
        (
            row["multiplicity"],
            row["K"],
            row["estimator"],
        )
        for row in rows
    })

    for m, K, method in keys:
        group = [
            row
            for row in rows
            if row["multiplicity"] == m
            and row["K"] == K
            and row["estimator"] == method
        ]

        valid = [
            row
            for row in group
            if row["valid"]
        ]

        times = [
            row["runtime_sec"]
            for row in group
        ]

        errors = [
            row["absolute_error_to_lambda_inf"]
            for row in valid
        ]

        summary.append({
            "multiplicity":
                m,
            "K":
                K,
            "paper_aligned_K":
                K in (10, 20),
            "estimator":
                method,
            "n_cases":
                len(group),
            "n_valid":
                len(valid),
            "valid_rate":
                (
                    len(valid)
                    / len(group)
                )
                if group
                else math.nan,
            "mean_absolute_error_to_lambda_inf":
                mean(errors),
            "median_absolute_error_to_lambda_inf":
                median(errors),
            "max_absolute_error_to_lambda_inf":
                max(errors)
                if errors
                else math.nan,
            "median_runtime_sec":
                median(times),
            "mean_runtime_sec":
                mean(times),
            "min_runtime_sec":
                min(times)
                if times
                else math.nan,
            "max_runtime_sec":
                max(times)
                if times
                else math.nan,
        })

    # Overall by method across all multiplicities and K values.
    for method in METHODS:
        group = [
            row
            for row in rows
            if row["estimator"]
            == method
        ]

        valid = [
            row
            for row in group
            if row["valid"]
        ]

        times = [
            row["runtime_sec"]
            for row in group
        ]

        errors = [
            row["absolute_error_to_lambda_inf"]
            for row in valid
        ]

        summary.append({
            "multiplicity":
                "ALL",
            "K":
                "ALL",
            "paper_aligned_K":
                False,
            "estimator":
                method,
            "n_cases":
                len(group),
            "n_valid":
                len(valid),
            "valid_rate":
                (
                    len(valid)
                    / len(group)
                )
                if group
                else math.nan,
            "mean_absolute_error_to_lambda_inf":
                mean(errors),
            "median_absolute_error_to_lambda_inf":
                median(errors),
            "max_absolute_error_to_lambda_inf":
                max(errors)
                if errors
                else math.nan,
            "median_runtime_sec":
                median(times),
            "mean_runtime_sec":
                mean(times),
            "min_runtime_sec":
                min(times)
                if times
                else math.nan,
            "max_runtime_sec":
                max(times)
                if times
                else math.nan,
        })

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--multiplicities",
        nargs="+",
        type=int,
        default=DEFAULT_MULTIPLICITIES,
    )

    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[
            10,
            20,
            40,
            50,
        ],
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
    )

    parser.add_argument(
        "--cutoff",
        default="1e-6",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/"
            "experiment4_nonlinear_repeated_newton_runtime"
        ),
    )

    args = parser.parse_args()

    if (
        not args.multiplicities
        or min(
            args.multiplicities
        ) < 2
    ):
        parser.error(
            "multiplicities must be >= 2"
        )

    if (
        not args.windows
        or min(
            args.windows
        ) < 2
    ):
        parser.error(
            "K values must be >= 2"
        )

    if args.replicates < 1:
        parser.error(
            "replicates must be >= 1"
        )

    if args.max_steps < 1:
        parser.error(
            "max-steps must be >= 1"
        )

    mp.mp.dps = args.dps
    cutoff = mp.mpf(
        str(args.cutoff)
    )

    if cutoff <= 0:
        parser.error(
            "cutoff must be positive"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Warm-up is deliberately outside all reported timings.
    warm_up()

    points = common_initial_points(
        seed=args.seed,
        replicates=args.replicates,
    )

    rows = []

    for m in args.multiplicities:
        lam_inf = float(
            lambda_inf_true(m)
        )

        for replicate, x0 in enumerate(
            points,
            start=1,
        ):
            trajectory = generate_until_cutoff(
                x0=x0,
                m=m,
                cutoff=cutoff,
                max_steps=args.max_steps,
            )

            last_retained = (
                len(trajectory) - 2
            )

            if mp.fabs(
                trajectory[last_retained]
                - L_TRUE
            ) < cutoff:
                raise AssertionError(
                    "Last retained point "
                    "violates cutoff."
                )

            if mp.fabs(
                trajectory[
                    last_retained + 1
                ]
                - L_TRUE
            ) >= cutoff:
                raise AssertionError(
                    "First excluded point "
                    "should be below cutoff."
                )

            for K in args.windows:
                latest = last_retained

                data_mp = trajectory[
                    latest - (K + 1):
                    latest + 1
                ]

                if len(data_mp) != K + 2:
                    raise RuntimeError(
                        f"m={m}, replicate={replicate}, "
                        f"K={K}: insufficient retained "
                        "observations."
                    )

                # Not timed.
                data = (
                    normalize_observed_record_to_float64(
                        data_mp
                    )
                )

                for method in METHODS:
                    try:
                        (
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
                        value = math.nan
                        valid = False
                        reason = (
                            f"{type(error).__name__}: "
                            f"{error}"
                        )
                        elapsed = 0.0

                    rows.append({
                        "multiplicity":
                            m,
                        "replicate":
                            replicate,
                        "x0":
                            float(x0),
                        "K":
                            K,
                        "paper_aligned_K":
                            K in (10, 20),
                        "last_retained_index":
                            last_retained,
                        "estimator":
                            method,
                        "lambda_inf_true":
                            lam_inf,
                        "lambda_hat":
                            value,
                        "absolute_error_to_lambda_inf":
                            (
                                abs(
                                    value
                                    - lam_inf
                                )
                                if valid
                                else math.nan
                            ),
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
                f"Completed m={m}, "
                f"replicate={replicate:02d}",
                flush=True,
            )

    summary = summarize(
        rows
    )

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
            "nonlinear repeated-root Newton runtime benchmark",
        "function_family":
            "(x-2)^m (x^2+1)",
        "multiplicities":
            args.multiplicities,
        "lambda_inf":
            "log((m-1)/m)",
        "windows":
            args.windows,
        "paper_aligned_K_values":
            [10, 20],
        "replicates":
            args.replicates,
        "seed":
            args.seed,
        "cutoff":
            str(args.cutoff),
        "trajectory_precision_dps":
            args.dps,
        "estimator_numeric_representation":
            "float64",
        "normalization":
            (
                "(x_i - x_last) / "
                "max_j |x_j - x_last|, "
                "computed in mpmath before "
                "float64 conversion"
            ),
        "same_selected_windows_as_accuracy_experiment":
            True,
        "warmup":
            (
                "one unreported call "
                "per estimator"
            ),
        "timed_region":
            "estimator call only",
        "excluded_from_timing": [
            "trajectory generation",
            "cutoff selection",
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
        if (
            row["multiplicity"]
            == "ALL"
        ):
            continue

        print(
            f"m={row['multiplicity']} "
            f"K={row['K']:2d} "
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
        if (
            row["multiplicity"]
            != "ALL"
        ):
            continue

        print(
            f"{row['estimator']:<18} "
            f"{row['median_runtime_sec']:.6f}s"
        )


if __name__ == "__main__":
    main()
