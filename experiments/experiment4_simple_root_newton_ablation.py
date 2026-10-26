"""
Experiment 4C ablation: why the proposed estimator does not visibly diverge
to -infinity in simple-root Newton convergence.

This is a diagnostic experiment for the proposed estimator only.  It uses the
same two simple-root Newton cases as Experiment 4C:

    poly_simple: f(x) = (x-2)(x^2+1), L=2
    exp_simple:  f(x) = exp(x-1)-1,   L=1

For both cases Newton convergence is quadratic and U'(L)=0, hence the ordinary
asymptotic Lyapunov exponent is -infinity.

The ablation separates four questions:

1. Does the official implementation numerically alter/cap the proposed formula?
   We independently recompute the formula using raw mpmath logarithms and compare
   it with the official returned components.

2. Is fixed-point estimation the cause?
   We recompute the same formula with the exact root L (oracle-L diagnostic).

3. Is the FIE correction opposing the residual contraction term?
   We save ell_err and log|ID_FIE| separately.

4. Are long windows simply nonlocal?
   We include diagnostic short windows K={2,3,5}, in addition to the usual
   K={10,20,40,50}, and save the start/end distances to the root.

No clipping, epsilon floor, or artificial replacement value is used in the raw
or oracle calculations.  If an exact zero or invalid logarithm occurs, that
manual calculation is marked invalid rather than modified.

Run from repository root:

    python -m experiments.experiment4_simple_root_newton_ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import mpmath as mp

from src.estimators.proposed import proposed_le_estimator


FUNCTIONS = ("poly_simple", "exp_simple")
VARIANTS = ("Proposed-WB", "Proposed-Aitken")


def root_for(name):
    if name == "poly_simple":
        return mp.mpf("2")
    if name == "exp_simple":
        return mp.mpf("1")
    raise ValueError(name)


def function_label(name):
    if name == "poly_simple":
        return "(x-2)(x^2+1)"
    if name == "exp_simple":
        return "exp(x-1)-1"
    raise ValueError(name)


def paper_initial_rule(name):
    if name == "poly_simple":
        return mp.mpf("4.0e9"), mp.mpf("4.0e7")
    if name == "exp_simple":
        return mp.mpf("55"), mp.mpf("2")
    raise ValueError(name)


def newton_update(name, x):
    x = mp.mpf(x)

    if name == "poly_simple":
        # Newton map for f=(x-2)(x^2+1).
        denominator = 3*x**2 - 4*x + 1
        if denominator == 0:
            raise ZeroDivisionError("Polynomial Newton denominator is zero.")
        return 2 * (x**3 - x**2 + 1) / denominator

    if name == "exp_simple":
        # Newton map for f=exp(x-1)-1.
        return x - 1 + mp.exp(1 - x)

    raise ValueError(name)


def initial_points(name, seed, replicates):
    center, half_width = paper_initial_rule(name)
    stream_offset = 0 if name == "poly_simple" else 1000003
    rng = random.Random(seed + stream_offset)

    return [
        center + half_width * mp.mpf(str(rng.uniform(-1.0, 1.0)))
        for _ in range(replicates)
    ]


def generate_to_cutoff_and_beyond(name, x0, cutoff, max_depth, max_steps):
    """
    Generate to the first point below cutoff and then far enough to support
    latest(depth)=j_cut-1+depth for every requested depth.
    """
    L = root_for(name)
    x = [mp.mpf(x0)]
    j_cut = None

    for _ in range(max_steps):
        x.append(newton_update(name, x[-1]))

        if j_cut is None and mp.fabs(x[-1] - L) < cutoff:
            j_cut = len(x) - 1
            break

    if j_cut is None:
        raise RuntimeError(f"{name}: cutoff not reached.")

    target_latest = j_cut - 1 + max_depth

    while len(x) - 1 < target_latest:
        x.append(newton_update(name, x[-1]))

    return x, j_cut


def manual_limit_hat(data, K, variant):
    if variant == "Proposed-WB":
        return mp.mpf(data[K + 1]), True, ""

    if variant == "Proposed-Aitken":
        x_nm1 = mp.mpf(data[K - 1])
        x_n = mp.mpf(data[K])
        x_np1 = mp.mpf(data[K + 1])

        denominator = x_np1 - 2*x_n + x_nm1

        if denominator == 0:
            return mp.nan, False, "exact-zero Aitken denominator"

        L_hat = x_nm1 - (x_n - x_nm1)**2 / denominator

        if not mp.isfinite(L_hat):
            return mp.nan, False, "nonfinite Aitken limit"

        return L_hat, True, ""

    raise ValueError(variant)


def hill(values):
    """
    Raw Hill/LID calculation with no floors or clipping.

        H(V,w) = -[ mean log(v/w) ]^{-1},  w=max(V).

    Invalid exact zeros are rejected rather than replaced.
    """
    values = [mp.mpf(v) for v in values]

    if not values:
        return mp.nan, False, "empty Hill sample"

    if any((not mp.isfinite(v)) for v in values):
        return mp.nan, False, "nonfinite Hill sample"

    if any(v <= 0 for v in values):
        return mp.nan, False, "nonpositive Hill sample"

    w = max(values)

    if w <= 0 or not mp.isfinite(w):
        return mp.nan, False, "invalid Hill boundary"

    logs = [mp.log(v / w) for v in values]

    if any((not mp.isfinite(q)) for q in logs):
        return mp.nan, False, "nonfinite Hill logarithm"

    mean_log = mp.fsum(logs) / len(logs)

    if mean_log == 0:
        return mp.nan, False, "exact-zero Hill denominator"

    H = -1 / mean_log

    if not mp.isfinite(H):
        return mp.nan, False, "nonfinite Hill estimate"

    return H, True, ""


def raw_components(data, K, L_hat):
    """
    Independently recompute the proposed estimator:

        ell_err = (1/K) sum_{j=1}^K [1/j log(r_j/r_0)]

    and FIE from K residual inputs and their K one-step forwards:

        R  = |x_0,...,x_{K-1} - L_hat|
        FR = |x_1,...,x_K     - L_hat|
        ID_FIE = H(R,max R) / H(FR,max FR)

        lambda_hat = ell_err + log|ID_FIE|.

    There is no epsilon, clipping, floor, or cap.
    """
    L_hat = mp.mpf(L_hat)

    residuals = [
        mp.fabs(mp.mpf(v) - L_hat)
        for v in data
    ]

    required = residuals[:K + 1]

    if any((not mp.isfinite(r)) for r in required):
        return {
            "valid": False,
            "failure_reason": "nonfinite required residual",
        }

    if any(r <= 0 for r in required):
        return {
            "valid": False,
            "failure_reason": "nonpositive required residual",
        }

    r0 = required[0]

    lagwise = [
        mp.log(required[j] / r0) / j
        for j in range(1, K + 1)
    ]

    if any((not mp.isfinite(v)) for v in lagwise):
        return {
            "valid": False,
            "failure_reason": "nonfinite lagwise log-rate",
        }

    ell_err = mp.fsum(lagwise) / K

    R = residuals[0:K]
    FR = residuals[1:K + 1]

    H_R, ok_R, reason_R = hill(R)
    if not ok_R:
        return {
            "valid": False,
            "failure_reason": "R Hill: " + reason_R,
            "ell_err": ell_err,
        }

    H_FR, ok_FR, reason_FR = hill(FR)
    if not ok_FR:
        return {
            "valid": False,
            "failure_reason": "FR Hill: " + reason_FR,
            "ell_err": ell_err,
        }

    if H_FR == 0:
        return {
            "valid": False,
            "failure_reason": "exact-zero forward Hill estimate",
            "ell_err": ell_err,
        }

    id_fie = H_R / H_FR

    if id_fie == 0 or not mp.isfinite(id_fie):
        return {
            "valid": False,
            "failure_reason": "invalid ID_FIE",
            "ell_err": ell_err,
            "id_fie": id_fie,
        }

    log_id = mp.log(mp.fabs(id_fie))
    lambda_hat = ell_err + log_id

    if not mp.isfinite(lambda_hat):
        return {
            "valid": False,
            "failure_reason": "nonfinite final raw lambda",
            "ell_err": ell_err,
            "id_fie": id_fie,
            "log_abs_id_fie": log_id,
        }

    return {
        "valid": True,
        "failure_reason": "",
        "ell_err": ell_err,
        "id_fie": id_fie,
        "log_abs_id_fie": log_id,
        "lambda_hat": lambda_hat,
        "min_required_residual": min(required),
        "max_required_residual": max(required),
    }


def official_estimate(data, K, variant):
    limit_method = "next" if variant == "Proposed-WB" else "aitken"

    try:
        result = proposed_le_estimator(
            data,
            K=K,
            end_index=K,
            limit_method=limit_method,
            numeric_backend="mpmath",
        )

        value = result.get("lambda_hat", mp.nan)
        valid = bool(result.get("valid", False) and mp.isfinite(value))

        return result, value, valid, result.get("failure_reason", "")

    except Exception as exc:
        return {}, mp.nan, False, f"{type(exc).__name__}: {exc}"


def finite_or_nan(value):
    try:
        return value if mp.isfinite(value) else mp.nan
    except Exception:
        return mp.nan


def text_value(value):
    if isinstance(value, mp.mpf):
        if value == mp.ninf:
            return "-inf"
        if value == mp.inf:
            return "inf"
        return mp.nstr(value, 90)
    return value


def write_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows([
            {k: text_value(v) for k, v in row.items()}
            for row in rows
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[2, 3, 5, 10, 20, 40, 50],
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6],
    )
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--dps",
        type=int,
        default=800,
        help="High precision is needed because quadratic convergence squares errors.",
    )
    parser.add_argument("--cutoff", default="1e-6")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/experiment4_simple_root_newton_ablation"),
    )

    args = parser.parse_args()

    if not args.windows or min(args.windows) < 2:
        parser.error("all K values must be >= 2")

    if not args.depths or min(args.depths) < 0:
        parser.error("all depths must be >= 0")

    if args.replicates < 1:
        parser.error("replicates must be >= 1")

    mp.mp.dps = args.dps
    cutoff = mp.mpf(str(args.cutoff))

    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    trajectory_rows = []

    for function_name in FUNCTIONS:
        L = root_for(function_name)

        for replicate, x0 in enumerate(
            initial_points(function_name, args.seed, args.replicates),
            start=1,
        ):
            trajectory, j_cut = generate_to_cutoff_and_beyond(
                name=function_name,
                x0=x0,
                cutoff=cutoff,
                max_depth=max(args.depths),
                max_steps=args.max_steps,
            )

            for depth in args.depths:
                latest = j_cut - 1 + depth
                latest_distance = mp.fabs(trajectory[latest] - L)

                trajectory_rows.append({
                    "function": function_name,
                    "function_label": function_label(function_name),
                    "replicate": replicate,
                    "x0": x0,
                    "depth": depth,
                    "cutoff_crossing_index": j_cut,
                    "latest_index": latest,
                    "latest_distance_to_L": latest_distance,
                    "latest_is_exact_L": trajectory[latest] == L,
                })

                for K in args.windows:
                    n = latest - 1
                    t0 = n - K

                    if t0 < 0:
                        continue

                    data = trajectory[t0:latest + 1]

                    if len(data) != K + 2:
                        raise AssertionError("Expected exactly K+2 observations.")

                    start_distance = mp.fabs(data[0] - L)
                    end_distance = mp.fabs(data[K] - L)

                    # How many of the K+1 points used in ell_err are already
                    # inside the paper's convergence cutoff?
                    local_count = sum(
                        mp.fabs(data[j] - L) < cutoff
                        for j in range(K + 1)
                    )

                    oracle = raw_components(data, K, L)

                    for variant in VARIANTS:
                        manual_L, manual_L_valid, manual_L_reason = manual_limit_hat(
                            data, K, variant
                        )

                        raw = (
                            raw_components(data, K, manual_L)
                            if manual_L_valid
                            else {
                                "valid": False,
                                "failure_reason": manual_L_reason,
                            }
                        )

                        official, official_lambda, official_valid, official_reason = (
                            official_estimate(data, K, variant)
                        )

                        official_L = official.get("L_hat", mp.nan)
                        official_ell = official.get("ell_err_hat", mp.nan)
                        official_id = official.get("id_fie_hat", mp.nan)

                        raw_lambda = raw.get("lambda_hat", mp.nan)
                        raw_ell = raw.get("ell_err", mp.nan)
                        raw_id = raw.get("id_fie", mp.nan)

                        row = {
                            "function": function_name,
                            "function_label": function_label(function_name),
                            "replicate": replicate,
                            "depth": depth,
                            "K": K,
                            "diagnostic_short_K": K in (2, 3, 5),
                            "paper_standard_K": K in (10, 20),
                            "fixed_point_true": L,
                            "window_start_distance_to_L": start_distance,
                            "window_end_distance_to_L": end_distance,
                            "latest_distance_to_L": latest_distance,
                            "local_points_inside_1e-6": local_count,
                            "fraction_ell_points_inside_1e-6":
                                mp.mpf(local_count) / (K + 1),
                            "variant": variant,

                            "official_valid": official_valid,
                            "official_failure_reason": official_reason,
                            "official_L_hat": official_L,
                            "official_lambda_hat": official_lambda,
                            "official_ell_err": official_ell,
                            "official_id_fie": official_id,
                            "official_log_abs_id_fie":
                                mp.log(mp.fabs(official_id))
                                if (
                                    official_valid
                                    and mp.isfinite(official_id)
                                    and official_id != 0
                                )
                                else mp.nan,

                            "manual_L_valid": manual_L_valid,
                            "manual_L_failure_reason": manual_L_reason,
                            "manual_L_hat": manual_L,
                            "manual_L_abs_error":
                                mp.fabs(manual_L - L)
                                if manual_L_valid
                                else mp.nan,

                            "raw_valid": raw.get("valid", False),
                            "raw_failure_reason": raw.get("failure_reason", ""),
                            "raw_lambda_hat": raw_lambda,
                            "raw_ell_err": raw_ell,
                            "raw_id_fie": raw_id,
                            "raw_log_abs_id_fie":
                                raw.get("log_abs_id_fie", mp.nan),
                            "raw_min_required_residual":
                                raw.get("min_required_residual", mp.nan),
                            "raw_max_required_residual":
                                raw.get("max_required_residual", mp.nan),

                            "oracle_valid": oracle.get("valid", False),
                            "oracle_failure_reason":
                                oracle.get("failure_reason", ""),
                            "oracle_lambda_hat":
                                oracle.get("lambda_hat", mp.nan),
                            "oracle_ell_err":
                                oracle.get("ell_err", mp.nan),
                            "oracle_id_fie":
                                oracle.get("id_fie", mp.nan),
                            "oracle_log_abs_id_fie":
                                oracle.get("log_abs_id_fie", mp.nan),

                            # Exact numerical differences, with no threshold-based
                            # pass/fail classification.
                            "abs_official_minus_raw_lambda":
                                mp.fabs(official_lambda - raw_lambda)
                                if (
                                    official_valid
                                    and raw.get("valid", False)
                                )
                                else mp.nan,
                            "abs_official_minus_raw_ell":
                                mp.fabs(official_ell - raw_ell)
                                if (
                                    official_valid
                                    and raw.get("valid", False)
                                    and mp.isfinite(official_ell)
                                )
                                else mp.nan,
                            "abs_official_minus_raw_id":
                                mp.fabs(official_id - raw_id)
                                if (
                                    official_valid
                                    and raw.get("valid", False)
                                    and mp.isfinite(official_id)
                                )
                                else mp.nan,
                            "abs_official_L_minus_manual_L":
                                mp.fabs(official_L - manual_L)
                                if (
                                    official_valid
                                    and manual_L_valid
                                    and mp.isfinite(official_L)
                                )
                                else mp.nan,
                        }

                        rows.append(row)

            print(
                f"Completed {function_name}, replicate={replicate:02d}, "
                f"cutoff_index={j_cut}",
                flush=True,
            )

    write_csv(args.output / "ablation_detail.csv", rows)
    write_csv(args.output / "trajectory_depths.csv", trajectory_rows)

    config = {
        "experiment": "simple-root Newton proposed-estimator ablation",
        "purpose": [
            "test whether official implementation numerically caps raw formula",
            "separate ell_err and FIE contributions",
            "remove fixed-point estimation through oracle-L diagnostic",
            "test short local windows versus long nonlocal windows",
        ],
        "functions": {
            "poly_simple": "(x-2)(x^2+1)",
            "exp_simple": "exp(x-1)-1",
        },
        "lambda_inf": "-infinity",
        "windows": args.windows,
        "depths": args.depths,
        "cutoff": str(args.cutoff),
        "replicates": args.replicates,
        "seed": args.seed,
        "dps": args.dps,
        "raw_formula_uses_clipping": False,
        "raw_formula_uses_epsilon_floor": False,
        "oracle_L_is_competing_estimator": False,
        "oracle_L_role": "diagnostic only",
    }

    (args.output / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\nWrote:")
    print(args.output / "ablation_detail.csv")
    print(args.output / "trajectory_depths.csv")
    print(args.output / "config.json")


if __name__ == "__main__":
    main()
