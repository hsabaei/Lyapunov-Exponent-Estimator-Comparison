"""
Experiment 4B: nonlinear repeated-root Newton family.

Family:
    f_m(x) = (x - 2)^m (x^2 + 1),
    m in {2, 3, 5}.

The root L=2 has multiplicity m.  Newton's iteration is genuinely nonlinear
away from the root.  Writing e=x-2, the Newton error map is

    e_next
      = e * [(m+1)e^2 + 4m e + 5(m-1)]
            / [(m+2)e^2 + 4(m+1)e + 5m].

Hence

    U_m'(2) = (m-1)/m,

and the exact asymptotic Lyapunov exponent is

    lambda_inf(m) = log((m-1)/m).

Therefore:
    m=2: lambda_inf = log(1/2)
    m=3: lambda_inf = log(2/3)
    m=5: lambda_inf = log(4/5)

Protocol
--------
- Common reproducible initial points around 4.0e9 +/- 4.0e7.
- 10 replicates by default.
- The same initial points are reused for every m so multiplicity is the only
  systematic change.
- Following the convergence-order paper's root-finding protocol, observations
  closer than 1e-6 to the root are excluded.
- For each m and replicate, find the last retained observation satisfying
      |x_t - L| >= 1e-6.
- For K in {10,20,40,50}, give every estimator exactly the same most-recent
  K+2 retained scalar observations.
- K={10,20} are paper-aligned sample sizes; K={40,50} are extensions.

Primary reference
-----------------
The primary target is the exact asymptotic LE

    lambda_inf(m) = log((m-1)/m).

A finite-window derivative LE is saved only as a diagnostic and is never
supplied to an estimator.

Run from repository root:
    python -m experiments.experiment4_nonlinear_repeated_newton
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


L_TRUE = mp.mpf("2")
DEFAULT_MULTIPLICITIES = [2, 3, 5]


def newton_error_update(e, m):
    """Exact Newton error map for f_m(x)=(x-2)^m(x^2+1)."""
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
            "Newton error map denominator is zero."
        )

    return e * numerator / denominator


def newton_update(x, m):
    x = mp.mpf(x)
    return L_TRUE + newton_error_update(x - L_TRUE, m)


def newton_derivative(x, m):
    """
    Analytic derivative of the Newton update.

    Let e=x-2 and g=e^2+4e+5.  With
        D = (m+2)e^2 + 4(m+1)e + 5m,
    Newton's derivative can be written without singular powers of e as

        U'(x) = g * A / D^2,

    where
        A = (m-1)D + e D',
        D' = 2(m+2)e + 4(m+1).

    At e=0 this gives U'(2)=(m-1)/m exactly.
    """
    x = mp.mpf(x)
    m = int(m)
    e = x - L_TRUE

    g = e**2 + 4*e + 5

    D = (
        (m + 2) * e**2
        + 4 * (m + 1) * e
        + 5 * m
    )

    Dp = (
        2 * (m + 2) * e
        + 4 * (m + 1)
    )

    if D == 0:
        raise ZeroDivisionError(
            "Newton derivative denominator is zero."
        )

    A = (m - 1) * D + e * Dp
    return g * A / (D * D)


def lambda_inf_true(m):
    m = int(m)
    if m <= 1:
        raise ValueError("Repeated-root multiplicity must be >= 2.")
    return mp.log(mp.mpf(m - 1) / m)


def generate_until_cutoff(x0, m, cutoff, max_steps):
    """
    Generate through the first point whose distance to L is below cutoff.
    """
    x = [mp.mpf(x0)]

    for _ in range(max_steps):
        nxt = newton_update(x[-1], m)
        x.append(nxt)

        if mp.fabs(nxt - L_TRUE) < cutoff:
            return x

    raise RuntimeError(
        f"m={m}: trajectory did not reach cutoff "
        f"within {max_steps} Newton steps."
    )


def common_initial_points(seed, replicates):
    """
    Paper-inspired root-finding scale:
        4.0e9 +/- 4.0e7.

    The same draws are reused for every multiplicity.
    """
    rng = random.Random(seed)

    center = mp.mpf("4.0e9")
    half_width = mp.mpf("4.0e7")

    return [
        center
        + half_width * mp.mpf(str(rng.uniform(-1.0, 1.0)))
        for _ in range(replicates)
    ]


def finite_window_le(x, m, t0, n):
    """
    K-transition derivative LE on t=t0,...,n-1.
    Diagnostic only.
    """
    values = [
        mp.log(mp.fabs(newton_derivative(x[t], m)))
        for t in range(t0, n)
    ]

    if not values:
        return mp.nan

    return mp.fsum(values) / len(values)


def estimate(data, K):
    methods = [
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

    for name, fn in methods:
        started = time.perf_counter()

        try:
            result = fn()
            value = result.get("lambda_hat", mp.nan)

            try:
                finite = bool(mp.isfinite(value))
            except Exception:
                finite = math.isfinite(float(value))

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
                    "Estimator returned an invalid "
                    "or nonfinite result."
                )

        except Exception as error:
            result = {}
            value = mp.nan
            valid = False
            reason = (
                f"{type(error).__name__}: "
                f"{error}"
            )

        elapsed = time.perf_counter() - started

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

        errors_inf = [
            row["absolute_error_to_lambda_inf"]
            for row in valid
        ]

        errors_ft = [
            row["absolute_error_to_lambda_ft"]
            for row in valid
        ]

        summary.append({
            "multiplicity": m,
            "asymptotic_multiplier_true":
                mp.mpf(m - 1) / m,
            "lambda_inf_true":
                lambda_inf_true(m),
            "K": K,
            "paper_aligned_K":
                K in (10, 20),
            "estimator": method,
            "total": len(group),
            "valid": len(valid),
            "invalid":
                len(group) - len(valid),
            "valid_rate":
                mp.mpf(len(valid)) / len(group)
                if group else mp.nan,
            "mean_absolute_error_to_lambda_inf":
                mp.fsum(errors_inf) / len(errors_inf)
                if errors_inf else mp.nan,
            "median_absolute_error_to_lambda_inf":
                median(errors_inf),
            "max_absolute_error_to_lambda_inf":
                max(errors_inf)
                if errors_inf else mp.nan,
            "mean_absolute_error_to_lambda_ft":
                mp.fsum(errors_ft) / len(errors_ft)
                if errors_ft else mp.nan,
            "correct_sign_fraction":
                mp.mpf(
                    sum(
                        row["correct_sign"]
                        for row in group
                    )
                ) / len(group)
                if group else mp.nan,
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
    )

    parser.add_argument(
        "--cutoff",
        default="1e-6",
        help="Root-distance cutoff inherited from the paper-style protocol.",
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
            "experiment4_nonlinear_repeated_newton"
        ),
    )

    args = parser.parse_args()

    if (
        not args.multiplicities
        or min(args.multiplicities) < 2
    ):
        parser.error(
            "all multiplicities must be >= 2"
        )

    if (
        not args.windows
        or min(args.windows) < 2
    ):
        parser.error(
            "all K values must be >= 2"
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

    cutoff = mp.mpf(str(args.cutoff))

    if cutoff <= 0:
        parser.error(
            "cutoff must be positive"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    points = common_initial_points(
        seed=args.seed,
        replicates=args.replicates,
    )

    rows = []
    trajectory_diagnostics = []

    for m in args.multiplicities:
        lam_inf = lambda_inf_true(m)
        multiplier = mp.mpf(m - 1) / m

        for replicate, x0 in enumerate(
            points,
            start=1,
        ):
            x = generate_until_cutoff(
                x0=x0,
                m=m,
                cutoff=cutoff,
                max_steps=args.max_steps,
            )

            # x[-1] is the first point below cutoff.
            last_retained = len(x) - 2

            if mp.fabs(
                x[last_retained] - L_TRUE
            ) < cutoff:
                raise AssertionError(
                    "Last retained observation violates cutoff."
                )

            if mp.fabs(
                x[last_retained + 1] - L_TRUE
            ) >= cutoff:
                raise AssertionError(
                    "First excluded observation should be below cutoff."
                )

            if last_retained + 1 < max(args.windows) + 2:
                raise RuntimeError(
                    f"m={m}, replicate={replicate}: "
                    "not enough retained observations "
                    "for largest K."
                )

            trajectory_diagnostics.append({
                "multiplicity": m,
                "replicate": replicate,
                "x0": x0,
                "fixed_point_true": L_TRUE,
                "asymptotic_multiplier_true":
                    multiplier,
                "lambda_inf_true":
                    lam_inf,
                "cutoff":
                    cutoff,
                "last_retained_index":
                    last_retained,
                "last_retained_x":
                    x[last_retained],
                "last_retained_distance":
                    mp.fabs(
                        x[last_retained]
                        - L_TRUE
                    ),
                "first_excluded_index":
                    last_retained + 1,
                "first_excluded_distance":
                    mp.fabs(
                        x[last_retained + 1]
                        - L_TRUE
                    ),
                "total_generated_observations":
                    len(x),
            })

            for K in args.windows:
                latest = last_retained
                n = latest - 1
                t0 = n - K

                if t0 < 0:
                    raise RuntimeError(
                        f"m={m}, replicate={replicate}, K={K}: "
                        "insufficient retained observations."
                    )

                data = x[t0:latest + 1]

                if len(data) != K + 2:
                    raise AssertionError(
                        "Expected exactly K+2 observations."
                    )

                lam_ft = finite_window_le(
                    x=x,
                    m=m,
                    t0=t0,
                    n=n,
                )

                window_start_distance = (
                    mp.fabs(
                        x[t0] - L_TRUE
                    )
                )

                window_end_distance = (
                    mp.fabs(
                        x[n] - L_TRUE
                    )
                )

                latest_distance = (
                    mp.fabs(
                        x[latest] - L_TRUE
                    )
                )

                # A simple dimensionless diagnostic of how far the window is
                # from its asymptotic local-linear regime:
                #
                # max_t |U'(x_t)-U'(L)| / |U'(L)|
                derivative_relative_deviation = max(
                    mp.fabs(
                        newton_derivative(x[t], m)
                        - multiplier
                    )
                    / mp.fabs(multiplier)
                    for t in range(t0, n)
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
                    rows.append({
                        "multiplicity":
                            m,
                        "replicate":
                            replicate,
                        "x0":
                            x0,
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
                        "window_start_distance_to_L":
                            window_start_distance,
                        "window_end_distance_to_L":
                            window_end_distance,
                        "latest_distance_to_L":
                            latest_distance,
                        "max_relative_derivative_deviation":
                            derivative_relative_deviation,
                        "estimator":
                            method,
                        "asymptotic_multiplier_true":
                            multiplier,
                        "lambda_inf_true":
                            lam_inf,
                        "lambda_ft_diagnostic":
                            lam_ft,
                        "lambda_hat":
                            value,
                        "absolute_error_to_lambda_inf":
                            mp.fabs(
                                value - lam_inf
                            )
                            if valid
                            else mp.nan,
                        "absolute_error_to_lambda_ft":
                            mp.fabs(
                                value - lam_ft
                            )
                            if valid
                            else mp.nan,
                        "correct_sign":
                            bool(
                                valid
                                and value < 0
                            ),
                        "valid":
                            valid,
                        "failure_reason":
                            reason,
                        "L_hat":
                            result.get(
                                "L_hat",
                                mp.nan,
                            ),
                        "limit_absolute_error":
                            mp.fabs(
                                result["L_hat"]
                                - L_TRUE
                            )
                            if (
                                valid
                                and "L_hat" in result
                            )
                            else mp.nan,
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
                f"Completed m={m}, "
                f"replicate={replicate:02d}, "
                f"x0={mp.nstr(x0, 12)}, "
                f"last_retained={last_retained}",
                flush=True,
            )

    summary = summarize(rows)

    write_csv(
        args.output / "estimates.csv",
        rows,
    )

    write_csv(
        args.output / "trajectory_diagnostics.csv",
        trajectory_diagnostics,
    )

    write_csv(
        args.output / "final_summary.csv",
        summary,
    )

    config = {
        "experiment":
            "nonlinear repeated-root Newton family",
        "function_family":
            "(x-2)^m (x^2+1)",
        "multiplicities":
            args.multiplicities,
        "fixed_point_true":
            "2",
        "newton_error_map":
            (
                "e_next = e * "
                "[(m+1)e^2 + 4m e + 5(m-1)] / "
                "[(m+2)e^2 + 4(m+1)e + 5m]"
            ),
        "asymptotic_multiplier":
            "(m-1)/m",
        "lambda_inf":
            "log((m-1)/m)",
        "exact_references": {
            str(m): {
                "multiplier":
                    mp.nstr(
                        mp.mpf(m - 1) / m,
                        40,
                    ),
                "lambda_inf":
                    mp.nstr(
                        lambda_inf_true(m),
                        70,
                    ),
            }
            for m in args.multiplicities
        },
        "initial_point_rule":
            "common draws from 4.0e9 +/- 4.0e7",
        "same_initial_points_across_m":
            True,
        "cutoff_used":
            mp.nstr(cutoff, 30),
        "paper_aligned_K_values":
            [10, 20],
        "extended_K_values": [
            K
            for K in args.windows
            if K not in (10, 20)
        ],
        "replicates":
            args.replicates,
        "seed":
            args.seed,
        "precision_dps":
            args.dps,
        "max_steps":
            args.max_steps,
        "primary_reference":
            "asymptotic lambda_inf = log((m-1)/m)",
        "finite_window_derivative_LE":
            "saved as diagnostic only",
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

    print("\n=== SUMMARY ===")

    for row in summary:
        print(
            f"m={row['multiplicity']} "
            f"K={row['K']:2d} "
            f"{row['estimator']:<18} "
            f"valid={row['valid']}/{row['total']} "
            f"MAE_inf="
            f"{mp.nstr(row['mean_absolute_error_to_lambda_inf'], 8)}"
        )


if __name__ == "__main__":
    main()
