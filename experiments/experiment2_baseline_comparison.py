"""
Experiment 2: complete baseline comparison on nonlinear stable fixed-point maps.

System
------
    x_{t+1} - L = a (x_t - L) + beta (x_t - L)^2

The fixed point is L and, for a != 0,

    lambda_true = log(|a|)

because U'(L) = a.

Nonlinearity design
-------------------
Let d = |x0 - L|. For eta in {0, 0.4, 0.9},

    |beta| = eta * (1 - |a|) / (2 d)

so on |x-L| <= d,

    |U'(x)| <= |a| + 2 |beta| d
             = |a| + eta (1 - |a|)
             = q_bound < 1.

For eta=0 there is one beta=0 case.
For eta>0 both beta signs are tested.

Thus each a has 5 nonlinear-strength/sign configurations:
    eta=0, beta=0
    eta=0.4, beta<0
    eta=0.4, beta>0
    eta=0.9, beta<0
    eta=0.9, beta>0

Comparison
----------
For each (a, beta, K), every method receives the same final K+2 observations:

    x_{N-K}, ..., x_N, x_{N+1}

The five reported estimators are:
    Proposed-Next
    Proposed-Aitken
    Yang-Wu
    Sun-Wu
    Rosenstein

The primary evaluation target is lambda_true = log(|a|).
No finite-window derivative oracle is used in the main comparison.

Important interpretation
------------------------
The nonlinear benchmark is quadratic. Yang-Wu uses a second-order local mapping,
so this experiment is structurally favorable to Yang-Wu. Do not use Experiment 2
alone to make broad claims about nonlinear systems. A later smooth non-quadratic
benchmark is needed before final comparative conclusions.

Numerics
--------
Trajectories are generated with mpmath at MP_DPS digits.
Proposed, Yang-Wu, and Rosenstein use their mpmath backends.
Sun-Wu preserves the high-precision relative geometry before its internally
normalized float64 RBF fit.

Runtime recorded here is diagnostic only and is NOT a fair cross-method runtime
comparison because arithmetic backends differ. A separate float64 timing study
can be run after accuracy/validity are checked.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from mpmath import mp

from src.estimators.proposed import proposed_le_estimator
from src.estimators.yang_wu import yang_wu_scalar_le_estimator
from src.estimators.sun_wu import sun_wu_scalar_rbf_le_estimator
from src.estimators.rosenstein import rosenstein_le_estimator


MP_DPS = 200

L = mp.mpf("2.0")
X0 = mp.mpf("1.3578696273508724")
N = 200

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
ETA_VALUES = ["0.0", "0.4", "0.9"]

OUTPUT_DIR = Path("results/tables")
DETAIL_PATH = OUTPUT_DIR / "experiment2_baseline_comparison.csv"
SUMMARY_PATH = OUTPUT_DIR / "experiment2_baseline_comparison_summary.csv"


def _as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return math.nan


def _safe_error(estimate, truth, valid):
    if not valid:
        return math.nan
    try:
        err = mp.fabs(mp.mpf(str(estimate)) - truth)
        return float(err)
    except Exception:
        return math.nan


def nonlinear_beta_cases(a):
    """
    Return tuples:
        (eta, beta_sign, beta, q_bound)

    beta_sign is 0 for beta=0, otherwise -1 or +1.
    """
    a = mp.mpf(a)
    d = mp.fabs(X0 - L)

    cases = []

    # eta = 0: only one beta=0 case.
    eta0 = mp.mpf("0.0")
    cases.append(
        (
            eta0,
            0,
            mp.mpf("0.0"),
            mp.fabs(a),
        )
    )

    # eta > 0: both beta signs.
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
            cases.append(
                (
                    eta,
                    beta_sign,
                    mp.mpf(beta_sign) * beta_abs,
                    q_bound,
                )
            )

    return cases


def generate_nonlinear_trajectory(a, beta):
    """
    Generate x_0, ..., x_N, x_{N+1}.
    """
    a = mp.mpf(a)
    beta = mp.mpf(beta)

    x = [X0]

    for _ in range(N + 1):
        e = x[-1] - L
        x_next = L + a * e + beta * e * e
        x.append(x_next)

    return x


def final_observation_record(x, K):
    """
    Same final K+2 observations for all five methods:
        x_{N-K}, ..., x_N, x_{N+1}
    """
    return x[N - K : N + 2]


def _base_row(
    *,
    a,
    eta,
    beta_sign,
    beta,
    q_bound,
    K,
    estimator,
    truth,
    result,
):
    estimate = result.get("lambda_hat")
    valid = bool(result.get("valid", False))

    return {
        "experiment": "quadratic_nonlinear",
        "a": float(a),
        "eta": float(eta),
        "beta_sign": beta_sign,
        "beta": float(beta),
        "q_bound": float(q_bound),
        "K": K,
        "N": N,
        "L": float(L),
        "x0": float(X0),
        "estimator": estimator,
        "lambda_true": float(truth),
        "lambda_hat": _as_float(estimate),
        "abs_error": _safe_error(
            estimate=estimate,
            truth=truth,
            valid=valid,
        ),
        "valid": valid,
        "failure_reason": result.get("failure_reason"),
        "runtime_sec_diagnostic_only": _as_float(
            result.get("runtime_sec")
        ),

        # Proposed-specific diagnostics.
        "L_hat": _as_float(result.get("L_hat")),
        "ell_err_hat": _as_float(result.get("ell_err_hat")),
        "id_fie_hat": _as_float(result.get("id_fie_hat")),

        # Yang-Wu diagnostics.
        "n_valid_local_maps": result.get("n_valid_local_maps"),
        "n_attempted_local_maps": result.get(
            "n_attempted_local_maps"
        ),

        # Sun-Wu diagnostics.
        "selected_hidden_nodes": result.get(
            "selected_hidden_nodes"
        ),
        "selected_width_neighbors": result.get(
            "selected_width_neighbors"
        ),
        "selected_cv_mse": _as_float(
            result.get("selected_cv_mse")
        ),
        "training_mse": _as_float(
            result.get("training_mse")
        ),

        # Rosenstein diagnostics.
        "n_neighbor_pairs": result.get("n_neighbor_pairs"),
        "n_horizons": (
            len(result.get("horizons", []))
            if result.get("horizons") is not None
            else None
        ),
    }


def run_case(
    *,
    a,
    eta,
    beta_sign,
    beta,
    q_bound,
    K,
    x,
):
    truth = mp.log(mp.fabs(a))

    data = final_observation_record(
        x=x,
        K=K,
    )

    rows = []

    proposed_next = proposed_le_estimator(
        x=data,
        K=K,
        end_index=K,
        limit_method="next",
        numeric_backend="mpmath",
    )

    rows.append(
        _base_row(
            a=a,
            eta=eta,
            beta_sign=beta_sign,
            beta=beta,
            q_bound=q_bound,
            K=K,
            estimator="Proposed-Next",
            truth=truth,
            result=proposed_next,
        )
    )

    proposed_aitken = proposed_le_estimator(
        x=data,
        K=K,
        end_index=K,
        limit_method="aitken",
        numeric_backend="mpmath",
    )

    rows.append(
        _base_row(
            a=a,
            eta=eta,
            beta_sign=beta_sign,
            beta=beta,
            q_bound=q_bound,
            K=K,
            estimator="Proposed-Aitken",
            truth=truth,
            result=proposed_aitken,
        )
    )

    yang_wu = yang_wu_scalar_le_estimator(
        x=data,
        evolution=1,
        r_min=None,
        numeric_backend="mpmath",
    )

    rows.append(
        _base_row(
            a=a,
            eta=eta,
            beta_sign=beta_sign,
            beta=beta,
            q_bound=q_bound,
            K=K,
            estimator="Yang-Wu",
            truth=truth,
            result=yang_wu,
        )
    )

    sun_wu = sun_wu_scalar_rbf_le_estimator(
        x=data,
        max_hidden_nodes=20,
        width_neighbors=None,
    )

    rows.append(
        _base_row(
            a=a,
            eta=eta,
            beta_sign=beta_sign,
            beta=beta,
            q_bound=q_bound,
            K=K,
            estimator="Sun-Wu",
            truth=truth,
            result=sun_wu,
        )
    )

    rosenstein = rosenstein_le_estimator(
        x=data,
        embedding_dim=1,
        lag=1,
        temporal_exclusion=0,
        sample_time=mp.mpf("1.0"),
        max_horizon=None,
        numeric_backend="mpmath",
    )

    rows.append(
        _base_row(
            a=a,
            eta=eta,
            beta_sign=beta_sign,
            beta=beta,
            q_bound=q_bound,
            K=K,
            estimator="Rosenstein",
            truth=truth,
            result=rosenstein,
        )
    )

    return rows


def _summary_one_group(scope, eta_label, K_label, estimator, subset):
    valid_rows = [
        row
        for row in subset
        if row["valid"]
        and math.isfinite(row["abs_error"])
    ]

    errors = [
        row["abs_error"]
        for row in valid_rows
    ]

    if errors:
        mean_error = statistics.fmean(errors)
        median_error = statistics.median(errors)
        max_error = max(errors)
    else:
        mean_error = math.nan
        median_error = math.nan
        max_error = math.nan

    sign_correct = 0

    for row in valid_rows:
        estimate = row["lambda_hat"]
        truth = row["lambda_true"]

        if (
            math.isfinite(estimate)
            and estimate * truth > 0
        ):
            sign_correct += 1

    return {
        "scope": scope,
        "eta": eta_label,
        "K": K_label,
        "estimator": estimator,
        "n_cases": len(subset),
        "n_valid": len(valid_rows),
        "valid_rate": (
            len(valid_rows) / len(subset)
            if subset
            else math.nan
        ),
        "mean_abs_error": mean_error,
        "median_abs_error": median_error,
        "max_abs_error": max_error,
        "sign_correct_rate": (
            sign_correct / len(valid_rows)
            if valid_rows
            else math.nan
        ),
    }


def summarize(rows):
    estimators = sorted(
        {row["estimator"] for row in rows}
    )

    summary = []

    # Overall.
    for estimator in estimators:
        subset = [
            row
            for row in rows
            if row["estimator"] == estimator
        ]

        summary.append(
            _summary_one_group(
                scope="overall",
                eta_label="ALL",
                K_label="ALL",
                estimator=estimator,
                subset=subset,
            )
        )

    # By nonlinearity strength eta.
    for eta in [0.0, 0.4, 0.9]:
        for estimator in estimators:
            subset = [
                row
                for row in rows
                if row["estimator"] == estimator
                and row["eta"] == eta
            ]

            summary.append(
                _summary_one_group(
                    scope="by_eta",
                    eta_label=eta,
                    K_label="ALL",
                    estimator=estimator,
                    subset=subset,
                )
            )

    # By K.
    for K in K_VALUES:
        for estimator in estimators:
            subset = [
                row
                for row in rows
                if row["estimator"] == estimator
                and row["K"] == K
            ]

            summary.append(
                _summary_one_group(
                    scope="by_K",
                    eta_label="ALL",
                    K_label=K,
                    estimator=estimator,
                    subset=subset,
                )
            )

    return summary


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
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    mp.dps = MP_DPS

    rows = []

    total_system_window_cases = (
        len(A_VALUES)
        * 5
        * len(K_VALUES)
    )

    completed = 0

    for a_text in A_VALUES:
        a = mp.mpf(a_text)

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
                rows.extend(
                    run_case(
                        a=a,
                        eta=eta,
                        beta_sign=beta_sign,
                        beta=beta,
                        q_bound=q_bound,
                        K=K,
                        x=x,
                    )
                )

                completed += 1

                print(
                    f"completed "
                    f"{completed}/"
                    f"{total_system_window_cases} "
                    f"system-window cases "
                    f"(a={float(a):+.2f}, "
                    f"eta={float(eta):.1f}, "
                    f"beta_sign={beta_sign:+d}, "
                    f"K={K})"
                )

    summary_rows = summarize(rows)

    write_csv(
        DETAIL_PATH,
        rows,
    )

    write_csv(
        SUMMARY_PATH,
        summary_rows,
    )

    print(
        "\n=== EXPERIMENT 2 "
        "NONLINEAR BASELINE COMPARISON ==="
    )

    print("\nOverall:")
    for row in summary_rows:
        if row["scope"] != "overall":
            continue

        print(
            f"{row['estimator']:<18} "
            f"valid={row['n_valid']}/"
            f"{row['n_cases']} "
            f"mean_abs_error="
            f"{row['mean_abs_error']:.6e} "
            f"median_abs_error="
            f"{row['median_abs_error']:.6e} "
            f"max_abs_error="
            f"{row['max_abs_error']:.6e} "
            f"sign_correct="
            f"{row['sign_correct_rate']:.3f}"
        )

    print("\nBy eta:")
    for eta in [0.0, 0.4, 0.9]:
        print(f"\neta={eta:.1f}")

        for row in summary_rows:
            if (
                row["scope"] == "by_eta"
                and row["eta"] == eta
            ):
                print(
                    f"  {row['estimator']:<18} "
                    f"valid={row['n_valid']}/"
                    f"{row['n_cases']} "
                    f"mean_abs_error="
                    f"{row['mean_abs_error']:.6e} "
                    f"median_abs_error="
                    f"{row['median_abs_error']:.6e}"
                )

    print("\nSaved:")
    print(DETAIL_PATH)
    print(SUMMARY_PATH)

    print(
        "\nInterpretation warning: "
        "this benchmark is quadratic, so the "
        "second-order Yang-Wu local model is "
        "structurally favored. Do not make broad "
        "nonlinear-method rankings from this "
        "experiment alone."
    )

    print(
        "\nNote: runtime_sec_diagnostic_only "
        "must not be used as a fair cross-method "
        "runtime comparison because the arithmetic "
        "backends differ."
    )


if __name__ == "__main__":
    main()
