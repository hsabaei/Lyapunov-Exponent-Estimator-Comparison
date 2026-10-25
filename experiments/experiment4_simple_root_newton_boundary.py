"""
Experiment 4C: simple-root Newton boundary experiment.

This experiment uses the two simple-root Newton-Raphson cases from the
convergence-order paper:

    1. f(x) = (x - 2)(x^2 + 1),  L = 2
    2. f(x) = exp(x - 1) - 1,    L = 1

Both roots are simple, so Newton's method is quadratically convergent and

    U'(L) = 0.

Therefore the ordinary asymptotic Lyapunov exponent is not finite:

    lambda_inf = log|U'(L)| = -infinity.

Consequently, this is NOT an absolute-error experiment against a finite
reference.  Instead, it tests boundary behavior as the observation window is
shifted progressively closer to the root.

Tail-depth protocol
-------------------
For each trajectory, first locate the convergence-order paper's root-distance
cutoff:

    |x_t - L| < 1e-6.

Let j_cut be the first iterate below that cutoff.  Define depth d in {0,1,2,3}
by choosing the extra/latest observation as

    latest(d) = j_cut - 1 + d.

Thus:
    d=0 : latest observation is the final point still above the paper cutoff;
    d=1 : first point below the paper cutoff;
    d=2 : one Newton step deeper;
    d=3 : two Newton steps deeper.

For each depth and each K in {10,20,40,50}, every estimator receives exactly
K+2 consecutive observations ending at latest(d).  A method consistent with
lambda_inf=-infinity should, in aggregate, produce increasingly negative LE
estimates as d increases.

A finite-window derivative average is recorded only as a diagnostic:

    lambda_FT = (1/K) sum log|U'(x_t)|.

It is not the asymptotic target and is never supplied to an estimator.

Run from repository root:

    python -m experiments.experiment4_simple_root_newton_boundary
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


FUNCTIONS = (
    "poly_simple",
    "exp_simple",
)

METHODS = (
    "Proposed-WB",
    "Proposed-Aitken",
    "Yang-Wu",
    "Sun-Wu",
    "Rosenstein",
)


def root_for(function_name):
    if function_name == "poly_simple":
        return mp.mpf("2")
    if function_name == "exp_simple":
        return mp.mpf("1")
    raise ValueError(function_name)


def function_label(function_name):
    if function_name == "poly_simple":
        return "(x-2)(x^2+1)"
    if function_name == "exp_simple":
        return "exp(x-1)-1"
    raise ValueError(function_name)


def paper_initial_rule(function_name):
    if function_name == "poly_simple":
        return (
            mp.mpf("4.0e9"),
            mp.mpf("4.0e7"),
        )

    if function_name == "exp_simple":
        return (
            mp.mpf("55"),
            mp.mpf("2"),
        )

    raise ValueError(function_name)


def newton_update(function_name, x):
    x = mp.mpf(x)

    if function_name == "poly_simple":
        # f=(x-2)(x^2+1)
        # U = 2(x^3-x^2+1)/(3x^2-4x+1)
        denominator = 3*x**2 - 4*x + 1

        if denominator == 0:
            raise ZeroDivisionError(
                "Polynomial Newton denominator is zero."
            )

        return (
            2 * (x**3 - x**2 + 1)
            / denominator
        )

    if function_name == "exp_simple":
        # Newton map for exp(x-1)-1:
        # U(x)=x-1+exp(1-x)
        return x - 1 + mp.exp(1 - x)

    raise ValueError(function_name)


def newton_derivative(function_name, x):
    x = mp.mpf(x)

    if function_name == "poly_simple":
        f = (x - 2) * (x*x + 1)
        fp = 3*x*x - 4*x + 1
        fpp = 6*x - 4

        if fp == 0:
            raise ZeroDivisionError(
                "Polynomial Newton derivative denominator is zero."
            )

        return f * fpp / (fp * fp)

    if function_name == "exp_simple":
        # U'(x)=1-exp(1-x), with U'(1)=0.
        return 1 - mp.exp(1 - x)

    raise ValueError(function_name)


def initial_points(function_name, seed, replicates):
    center, half_width = paper_initial_rule(function_name)

    # Separate deterministic stream for each function while preserving
    # reproducibility.
    offset = (
        0
        if function_name == "poly_simple"
        else 1000003
    )

    rng = random.Random(seed + offset)

    return [
        center
        + half_width
        * mp.mpf(str(rng.uniform(-1.0, 1.0)))
        for _ in range(replicates)
    ]


def generate_to_cutoff_and_beyond(
    function_name,
    x0,
    cutoff,
    extra_steps,
    max_steps,
):
    """
    Generate until the first point below cutoff, then continue extra_steps.
    Returns (trajectory, crossing_index).
    """
    L = root_for(function_name)
    x = [mp.mpf(x0)]
    crossing = None

    for _ in range(max_steps):
        nxt = newton_update(
            function_name,
            x[-1],
        )
        x.append(nxt)

        if (
            crossing is None
            and mp.fabs(nxt - L) < cutoff
        ):
            crossing = len(x) - 1
            break

    if crossing is None:
        raise RuntimeError(
            f"{function_name}: cutoff not reached "
            f"within {max_steps} Newton steps."
        )

    for _ in range(extra_steps):
        x.append(
            newton_update(
                function_name,
                x[-1],
            )
        )

    return x, crossing


def finite_window_le(
    function_name,
    x,
    t0,
    n,
):
    """
    K-transition derivative average on t=t0,...,n-1.
    Diagnostic only.
    """
    values = []

    for t in range(t0, n):
        derivative = mp.fabs(
            newton_derivative(
                function_name,
                x[t],
            )
        )

        if derivative == 0:
            return mp.ninf

        values.append(
            mp.log(derivative)
        )

    return mp.fsum(values) / len(values)


def estimate(data, K):
    estimators = [
        (
            "Proposed-WB",
            lambda: proposed_le_estimator(
                data,
                K=K,
                end_index=K,
                limit_method="next",
                numeric_backend="mpmath",
            ),
        ),
        (
            "Proposed-Aitken",
            lambda: proposed_le_estimator(
                data,
                K=K,
                end_index=K,
                limit_method="aitken",
                numeric_backend="mpmath",
            ),
        ),
        (
            "Yang-Wu",
            lambda: yang_wu_scalar_le_estimator(
                data,
                evolution=1,
                r_min=None,
                numeric_backend="mpmath",
            ),
        ),
        (
            "Sun-Wu",
            lambda: sun_wu_scalar_rbf_le_estimator(
                data,
                max_hidden_nodes=20,
                width_neighbors=None,
            ),
        ),
        (
            "Rosenstein",
            lambda: rosenstein_le_estimator(
                data,
                embedding_dim=1,
                lag=1,
                temporal_exclusion=0,
                sample_time=mp.mpf(1),
                max_horizon=None,
                numeric_backend="mpmath",
            ),
        ),
    ]

    for name, fn in estimators:
        started = time.perf_counter()

        try:
            result = fn()
            value = result.get(
                "lambda_hat",
                mp.nan,
            )

            try:
                finite = bool(
                    mp.isfinite(value)
                )
            except Exception:
                finite = math.isfinite(
                    float(value)
                )

            valid = bool(
                result.get("valid", False)
                and finite
            )

            reason = result.get(
                "failure_reason",
                "",
            )

            if not valid and not reason:
                reason = (
                    "Estimator returned invalid "
                    "or nonfinite output."
                )

        except Exception as error:
            result = {}
            value = mp.nan
            valid = False
            reason = (
                f"{type(error).__name__}: "
                f"{error}"
            )

        elapsed = (
            time.perf_counter()
            - started
        )

        yield (
            name,
            result,
            value,
            valid,
            reason,
            elapsed,
        )


def text_value(value):
    if isinstance(value, mp.mpf):
        if value == mp.ninf:
            return "-inf"
        if value == mp.inf:
            return "inf"
        return mp.nstr(value, 70)

    return value


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

        writer.writerows([
            {
                key: text_value(value)
                for key, value in row.items()
            }
            for row in rows
        ])


def median(values):
    values = sorted(values)
    n = len(values)

    if n == 0:
        return mp.nan

    if n % 2:
        return values[n // 2]

    return (
        values[n//2 - 1]
        + values[n//2]
    ) / 2


def summarize(rows):
    summary = []

    keys = sorted({
        (
            row["function"],
            row["depth"],
            row["K"],
            row["estimator"],
        )
        for row in rows
    })

    for function_name, depth, K, method in keys:
        group = [
            row
            for row in rows
            if row["function"] == function_name
            and row["depth"] == depth
            and row["K"] == K
            and row["estimator"] == method
        ]

        valid = [
            row
            for row in group
            if row["valid"]
        ]

        hats = [
            row["lambda_hat"]
            for row in valid
        ]

        ft_gaps = [
            row["absolute_difference_to_lambda_ft"]
            for row in valid
            if mp.isfinite(
                row["lambda_ft_diagnostic"]
            )
        ]

        ft_values = [
            row["lambda_ft_diagnostic"]
            for row in group
            if mp.isfinite(
                row["lambda_ft_diagnostic"]
            )
        ]

        summary.append({
            "function":
                function_name,
            "function_label":
                function_label(function_name),
            "depth":
                depth,
            "K":
                K,
            "paper_aligned_K":
                K in (10, 20),
            "estimator":
                method,
            "total":
                len(group),
            "valid":
                len(valid),
            "invalid":
                len(group) - len(valid),
            "valid_rate":
                mp.mpf(len(valid)) / len(group)
                if group else mp.nan,
            "negative_fraction":
                mp.mpf(
                    sum(
                        row["correct_negative_sign"]
                        for row in group
                    )
                ) / len(group)
                if group else mp.nan,
            "mean_lambda_hat":
                mp.fsum(hats) / len(hats)
                if hats else mp.nan,
            "median_lambda_hat":
                median(hats),
            "min_lambda_hat":
                min(hats)
                if hats else mp.nan,
            "max_lambda_hat":
                max(hats)
                if hats else mp.nan,
            "mean_lambda_ft_diagnostic":
                mp.fsum(ft_values) / len(ft_values)
                if ft_values else mp.nan,
            "mean_absolute_difference_to_lambda_ft":
                mp.fsum(ft_gaps) / len(ft_gaps)
                if ft_gaps else mp.nan,
        })

    return summary


def trend_summary(rows, depths):
    """
    For each function/K/method, check whether lambda_hat becomes strictly
    more negative as depth increases within each replicate.
    """
    output = []

    keys = sorted({
        (
            row["function"],
            row["K"],
            row["estimator"],
        )
        for row in rows
    })

    for function_name, K, method in keys:
        replicate_ids = sorted({
            row["replicate"]
            for row in rows
            if row["function"] == function_name
            and row["K"] == K
            and row["estimator"] == method
        })

        n_complete = 0
        n_strict = 0
        deltas = []

        for replicate in replicate_ids:
            selected = {
                row["depth"]: row
                for row in rows
                if row["function"] == function_name
                and row["K"] == K
                and row["estimator"] == method
                and row["replicate"] == replicate
            }

            if not all(
                d in selected
                and selected[d]["valid"]
                for d in depths
            ):
                continue

            values = [
                selected[d]["lambda_hat"]
                for d in depths
            ]

            n_complete += 1

            if all(
                values[i+1] < values[i]
                for i in range(
                    len(values) - 1
                )
            ):
                n_strict += 1

            deltas.append(
                values[-1]
                - values[0]
            )

        output.append({
            "function":
                function_name,
            "function_label":
                function_label(function_name),
            "K":
                K,
            "estimator":
                method,
            "n_complete_replicates":
                n_complete,
            "strictly_more_negative_fraction":
                (
                    mp.mpf(n_strict)
                    / n_complete
                    if n_complete
                    else mp.nan
                ),
            "mean_lambda_change_deepest_minus_shallowest":
                (
                    mp.fsum(deltas)
                    / len(deltas)
                    if deltas
                    else mp.nan
                ),
            "median_lambda_change_deepest_minus_shallowest":
                median(deltas),
        })

    return output


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[10, 20, 40, 50],
    )

    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
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
            "experiment4_simple_root_newton_boundary"
        ),
    )

    args = parser.parse_args()

    if (
        not args.windows
        or min(args.windows) < 2
    ):
        parser.error(
            "all K values must be >= 2"
        )

    if (
        not args.depths
        or min(args.depths) < 0
    ):
        parser.error(
            "depths must be nonnegative"
        )

    if sorted(set(args.depths)) != args.depths:
        parser.error(
            "depths must be unique and sorted"
        )

    if args.replicates < 1:
        parser.error(
            "replicates must be >= 1"
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

    max_depth = max(args.depths)
    # latest(d)=j_cut-1+d, so we need points through j_cut-1+max_depth.
    extra_steps_after_crossing = max(
        0,
        max_depth - 1,
    )

    rows = []
    diagnostics = []

    for function_name in FUNCTIONS:
        L = root_for(
            function_name
        )

        points = initial_points(
            function_name,
            args.seed,
            args.replicates,
        )

        for replicate, x0 in enumerate(
            points,
            start=1,
        ):
            trajectory, j_cut = (
                generate_to_cutoff_and_beyond(
                    function_name=function_name,
                    x0=x0,
                    cutoff=cutoff,
                    extra_steps=
                        extra_steps_after_crossing,
                    max_steps=args.max_steps,
                )
            )

            if mp.fabs(
                trajectory[j_cut] - L
            ) >= cutoff:
                raise AssertionError(
                    "Crossing point must be below cutoff."
                )

            if mp.fabs(
                trajectory[j_cut - 1] - L
            ) < cutoff:
                raise AssertionError(
                    "Previous point must be outside cutoff."
                )

            for depth in args.depths:
                latest = (
                    j_cut - 1 + depth
                )

                if latest >= len(
                    trajectory
                ):
                    raise RuntimeError(
                        "Insufficient generated depth."
                    )

                diagnostics.append({
                    "function":
                        function_name,
                    "function_label":
                        function_label(function_name),
                    "replicate":
                        replicate,
                    "x0":
                        x0,
                    "fixed_point_true":
                        L,
                    "lambda_inf_true":
                        "-inf",
                    "paper_cutoff":
                        cutoff,
                    "cutoff_crossing_index":
                        j_cut,
                    "depth":
                        depth,
                    "latest_observation_index":
                        latest,
                    "latest_observation":
                        trajectory[latest],
                    "latest_distance_to_L":
                        mp.fabs(
                            trajectory[latest]
                            - L
                        ),
                })

                for K in args.windows:
                    n = latest - 1
                    t0 = n - K

                    if t0 < 0:
                        raise RuntimeError(
                            f"{function_name}, "
                            f"replicate={replicate}, "
                            f"depth={depth}, K={K}: "
                            "insufficient observations."
                        )

                    data = trajectory[
                        t0:latest + 1
                    ]

                    if len(data) != K + 2:
                        raise AssertionError(
                            "Expected exactly K+2 observations."
                        )

                    lambda_ft = (
                        finite_window_le(
                            function_name,
                            trajectory,
                            t0,
                            n,
                        )
                    )

                    for (
                        method,
                        result,
                        value,
                        valid,
                        reason,
                        elapsed,
                    ) in estimate(
                        data,
                        K,
                    ):
                        ft_gap = mp.nan

                        if (
                            valid
                            and mp.isfinite(
                                lambda_ft
                            )
                        ):
                            ft_gap = mp.fabs(
                                value
                                - lambda_ft
                            )

                        rows.append({
                            "function":
                                function_name,
                            "function_label":
                                function_label(
                                    function_name
                                ),
                            "replicate":
                                replicate,
                            "x0":
                                x0,
                            "fixed_point_true":
                                L,
                            "lambda_inf_true":
                                "-inf",
                            "depth":
                                depth,
                            "K":
                                K,
                            "paper_aligned_K":
                                K in (10, 20),
                            "global_window_start":
                                t0,
                            "global_end_index":
                                n,
                            "global_latest_observation_index":
                                latest,
                            "latest_distance_to_L":
                                mp.fabs(
                                    trajectory[latest]
                                    - L
                                ),
                            "window_start_distance_to_L":
                                mp.fabs(
                                    trajectory[t0]
                                    - L
                                ),
                            "lambda_ft_diagnostic":
                                lambda_ft,
                            "estimator":
                                method,
                            "lambda_hat":
                                value,
                            "valid":
                                valid,
                            "correct_negative_sign":
                                bool(
                                    valid
                                    and value < 0
                                ),
                            "failure_reason":
                                reason,
                            "absolute_difference_to_lambda_ft":
                                ft_gap,
                            "L_hat":
                                result.get(
                                    "L_hat",
                                    mp.nan,
                                ),
                            "ell_err_hat":
                                result.get(
                                    "ell_err_hat",
                                    mp.nan,
                                ),
                            "id_fie_hat":
                                result.get(
                                    "id_fie_hat",
                                    mp.nan,
                                ),
                            "runtime_sec":
                                elapsed,
                        })

            print(
                f"Completed {function_name}, "
                f"replicate={replicate:02d}, "
                f"cutoff_index={j_cut}",
                flush=True,
            )

    summary = summarize(
        rows
    )

    trends = trend_summary(
        rows,
        args.depths,
    )

    write_csv(
        args.output
        / "estimates.csv",
        rows,
    )

    write_csv(
        args.output
        / "trajectory_diagnostics.csv",
        diagnostics,
    )

    write_csv(
        args.output
        / "final_summary.csv",
        summary,
    )

    write_csv(
        args.output
        / "trend_summary.csv",
        trends,
    )

    config = {
        "experiment":
            "simple-root Newton lambda=-infinity boundary",
        "functions": {
            "poly_simple": {
                "f":
                    "(x-2)(x^2+1)",
                "L":
                    "2",
                "paper_initial_point_rule":
                    "4.0e9 +/- 4.0e7",
            },
            "exp_simple": {
                "f":
                    "exp(x-1)-1",
                "L":
                    "1",
                "paper_initial_point_rule":
                    "55 +/- 2",
            },
        },
        "newton_convergence_order":
            2,
        "asymptotic_derivative":
            "U'(L)=0",
        "lambda_inf":
            "-infinity",
        "paper_cutoff":
            str(args.cutoff),
        "depth_definition":
            (
                "latest(depth) = cutoff_crossing_index "
                "- 1 + depth"
            ),
        "depths":
            args.depths,
        "windows":
            args.windows,
        "paper_aligned_K_values":
            [10, 20],
        "replicates":
            args.replicates,
        "seed":
            args.seed,
        "precision_dps":
            args.dps,
        "primary_evaluation":
            (
                "validity, negative sign, and movement "
                "toward more-negative estimates as depth increases"
            ),
        "finite_window_derivative_LE":
            "diagnostic only",
        "no_absolute_error_to_lambda_inf":
            True,
        "timing":
            (
                "single-call elapsed seconds; "
                "not a controlled runtime benchmark"
            ),
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

    print("\n=== BOUNDARY SUMMARY ===")

    for row in trends:
        print(
            f"{row['function']:<12} "
            f"K={row['K']:2d} "
            f"{row['estimator']:<18} "
            f"strict-more-negative="
            f"{mp.nstr(row['strictly_more_negative_fraction'], 5)} "
            f"delta="
            f"{mp.nstr(row['mean_lambda_change_deepest_minus_shallowest'], 8)}"
        )


if __name__ == "__main__":
    main()
