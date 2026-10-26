"""
Experiment 4C (final): simple-root Newton boundary/scope experiment.

Purpose
-------
Test the proposed Lyapunov estimator at the boundary U'(L)=0, where the
ordinary asymptotic Lyapunov exponent is -infinity.

Newton cases
------------
1) f(x) = (x-2)(x^2+1), L=2
2) f(x) = exp(x-1)-1,   L=1

Both are simple-root Newton iterations with quadratic local convergence.

Protocol
--------
The paper-style root-distance cutoff |x-L|<1e-6 defines a reference crossing
index j_cut.  For depth d, the latest observation is

    latest(d) = j_cut - 1 + d.

Depths d=0,...,6 move the window progressively deeper into the local
quadratic-convergence regime.

Window roles:
    K={2,3,5}: local asymptotic diagnostics
    K=10:      transition window
    K={20,40,50}: nonlocal-window controls

This experiment is intentionally proposed-estimator-only (Proposed-WB and
Proposed-Aitken).  Its goal is scope/boundary analysis, not another baseline
ranking.

Outputs
-------
- detail.csv
- final_boundary_summary.csv
- final_boundary_trend_summary.csv
- implementation_check_summary.csv
- config.json

The script also independently recomputes the estimator with raw mpmath logs,
without clipping/floors/caps, and computes an oracle-L diagnostic.  These are
diagnostics only, not competing estimators.

Run:
    python -m experiments.experiment4_simple_root_newton_final
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


def label_for(name):
    if name == "poly_simple":
        return "(x-2)(x^2+1)"
    if name == "exp_simple":
        return "exp(x-1)-1"
    raise ValueError(name)


def initial_rule(name):
    if name == "poly_simple":
        return mp.mpf("4e9"), mp.mpf("4e7")
    if name == "exp_simple":
        return mp.mpf("55"), mp.mpf("2")
    raise ValueError(name)


def window_role(K):
    if K in (2, 3, 5):
        return "local diagnostic"
    if K == 10:
        return "transition"
    return "nonlocal-window control"


def newton_update(name, x):
    x = mp.mpf(x)

    if name == "poly_simple":
        denominator = 3*x**2 - 4*x + 1
        if denominator == 0:
            raise ZeroDivisionError("Polynomial Newton denominator is zero.")
        return 2 * (x**3 - x**2 + 1) / denominator

    if name == "exp_simple":
        return x - 1 + mp.exp(1 - x)

    raise ValueError(name)


def initial_points(name, seed, replicates):
    center, half_width = initial_rule(name)
    offset = 0 if name == "poly_simple" else 1000003
    rng = random.Random(seed + offset)
    return [
        center + half_width * mp.mpf(str(rng.uniform(-1.0, 1.0)))
        for _ in range(replicates)
    ]


def generate(name, x0, cutoff, max_depth, max_steps):
    L = root_for(name)
    x = [mp.mpf(x0)]
    crossing = None

    for _ in range(max_steps):
        x.append(newton_update(name, x[-1]))
        if crossing is None and mp.fabs(x[-1] - L) < cutoff:
            crossing = len(x) - 1
            break

    if crossing is None:
        raise RuntimeError(f"{name}: cutoff not reached.")

    target_latest = crossing - 1 + max_depth
    while len(x) - 1 < target_latest:
        x.append(newton_update(name, x[-1]))

    return x, crossing


def hill(values):
    values = [mp.mpf(v) for v in values]
    if any((not mp.isfinite(v)) or v <= 0 for v in values):
        return mp.nan, False
    w = max(values)
    logs = [mp.log(v / w) for v in values]
    mean_log = mp.fsum(logs) / len(logs)
    if mean_log == 0 or not mp.isfinite(mean_log):
        return mp.nan, False
    H = -1 / mean_log
    return H, bool(mp.isfinite(H))


def manual_limit(data, K, variant):
    if variant == "Proposed-WB":
        return mp.mpf(data[K + 1]), True

    x_nm1 = mp.mpf(data[K - 1])
    x_n = mp.mpf(data[K])
    x_np1 = mp.mpf(data[K + 1])
    denom = x_np1 - 2*x_n + x_nm1

    if denom == 0:
        return mp.nan, False

    Lhat = x_nm1 - (x_n - x_nm1)**2 / denom
    return Lhat, bool(mp.isfinite(Lhat))


def raw_components(data, K, Lhat):
    residuals = [mp.fabs(mp.mpf(v) - Lhat) for v in data]
    needed = residuals[:K + 1]

    if any((not mp.isfinite(r)) or r <= 0 for r in needed):
        return {"valid": False}

    r0 = needed[0]
    lagwise = [mp.log(needed[j] / r0) / j for j in range(1, K + 1)]
    ell = mp.fsum(lagwise) / K

    HR, ok1 = hill(residuals[0:K])
    HFR, ok2 = hill(residuals[1:K + 1])

    if not ok1 or not ok2 or HFR == 0:
        return {"valid": False, "ell_err": ell}

    id_fie = HR / HFR
    if id_fie == 0 or not mp.isfinite(id_fie):
        return {"valid": False, "ell_err": ell}

    log_id = mp.log(mp.fabs(id_fie))
    lam = ell + log_id

    return {
        "valid": bool(mp.isfinite(lam)),
        "ell_err": ell,
        "id_fie": id_fie,
        "log_abs_id_fie": log_id,
        "lambda_hat": lam,
    }


def official(data, K, variant):
    method = "next" if variant == "Proposed-WB" else "aitken"
    result = proposed_le_estimator(
        data,
        K=K,
        end_index=K,
        limit_method=method,
        numeric_backend="mpmath",
    )
    value = result.get("lambda_hat", mp.nan)
    valid = bool(result.get("valid", False) and mp.isfinite(value))
    return result, value, valid


def median(values):
    values = sorted(values)
    n = len(values)
    if not n:
        return mp.nan
    return values[n//2] if n % 2 else (values[n//2-1] + values[n//2]) / 2


def text_value(v):
    if isinstance(v, mp.mpf):
        return mp.nstr(v, 90)
    return v


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows([{k: text_value(v) for k, v in r.items()} for r in rows])


def summarize(rows):
    output = []
    keys = sorted({
        (r["function"], r["function_label"], r["variant"], r["K"], r["depth"])
        for r in rows
    })

    for function, label, variant, K, depth in keys:
        g = [
            r for r in rows
            if r["function"] == function
            and r["variant"] == variant
            and r["K"] == K
            and r["depth"] == depth
        ]
        valid = [r for r in g if r["official_valid"]]
        output.append({
            "function": function,
            "function_label": label,
            "variant": variant,
            "K": K,
            "window_role": window_role(K),
            "depth": depth,
            "n": len(g),
            "valid": len(valid),
            "valid_rate": mp.mpf(len(valid))/len(g),
            "mean_lambda_hat": mp.fsum(r["official_lambda_hat"] for r in valid)/len(valid),
            "median_lambda_hat": median([r["official_lambda_hat"] for r in valid]),
            "mean_ell_err": mp.fsum(r["official_ell_err"] for r in valid)/len(valid),
            "mean_log_abs_id_fie": mp.fsum(r["official_log_abs_id_fie"] for r in valid)/len(valid),
            "mean_fraction_local": mp.fsum(r["fraction_local"] for r in g)/len(g),
            "mean_window_start_distance":
                mp.fsum(r["window_start_distance"] for r in g)/len(g),
            "mean_latest_distance":
                mp.fsum(r["latest_distance"] for r in g)/len(g),
            "mean_abs_official_minus_oracle":
                mp.fsum(mp.fabs(r["official_lambda_hat"]-r["oracle_lambda_hat"])
                        for r in valid if r["oracle_valid"])/len(valid),
            "max_abs_official_minus_raw":
                max(r["abs_official_minus_raw"] for r in valid),
        })
    return output


def trends(rows, depths):
    output = []
    keys = sorted({
        (r["function"], r["function_label"], r["variant"], r["K"])
        for r in rows
    })

    for function, label, variant, K in keys:
        reps = sorted({
            r["replicate"] for r in rows
            if r["function"] == function
            and r["variant"] == variant
            and r["K"] == K
        })
        complete = strict = 0
        changes = []

        for rep in reps:
            selected = {
                r["depth"]: r for r in rows
                if r["function"] == function
                and r["variant"] == variant
                and r["K"] == K
                and r["replicate"] == rep
            }
            if not all(d in selected and selected[d]["official_valid"] for d in depths):
                continue
            vals = [selected[d]["official_lambda_hat"] for d in depths]
            complete += 1
            strict += int(all(vals[i+1] < vals[i] for i in range(len(vals)-1)))
            changes.append(vals[-1]-vals[0])

        deepest = [
            r["fraction_local"] for r in rows
            if r["function"] == function
            and r["variant"] == variant
            and r["K"] == K
            and r["depth"] == max(depths)
        ]

        output.append({
            "function": function,
            "function_label": label,
            "variant": variant,
            "K": K,
            "window_role": window_role(K),
            "n_complete_replicates": complete,
            "strictly_more_negative_fraction":
                mp.mpf(strict)/complete if complete else mp.nan,
            "mean_lambda_change_deepest_minus_shallowest":
                mp.fsum(changes)/len(changes) if changes else mp.nan,
            "median_lambda_change_deepest_minus_shallowest":
                median(changes),
            "mean_fraction_local_at_deepest_depth":
                mp.fsum(deepest)/len(deepest) if deepest else mp.nan,
        })
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--windows", nargs="+", type=int,
                   default=[2, 3, 5, 10, 20, 40, 50])
    p.add_argument("--depths", nargs="+", type=int,
                   default=[0, 1, 2, 3, 4, 5, 6])
    p.add_argument("--replicates", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dps", type=int, default=800)
    p.add_argument("--cutoff", default="1e-6")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--output", type=Path,
                   default=Path("results/experiment4_simple_root_newton_final"))
    args = p.parse_args()

    mp.mp.dps = args.dps
    cutoff = mp.mpf(str(args.cutoff))
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []

    for name in FUNCTIONS:
        L = root_for(name)

        for rep, x0 in enumerate(initial_points(name, args.seed, args.replicates), start=1):
            trajectory, crossing = generate(
                name, x0, cutoff, max(args.depths), args.max_steps
            )

            for depth in args.depths:
                latest = crossing - 1 + depth

                for K in args.windows:
                    n = latest - 1
                    t0 = n - K
                    if t0 < 0:
                        continue

                    data = trajectory[t0:latest + 1]
                    if len(data) != K + 2:
                        raise AssertionError("Expected K+2 observations.")

                    local_count = sum(
                        mp.fabs(data[j] - L) < cutoff
                        for j in range(K + 1)
                    )
                    fraction_local = mp.mpf(local_count)/(K + 1)

                    oracle = raw_components(data, K, L)

                    for variant in VARIANTS:
                        Lhat, L_ok = manual_limit(data, K, variant)
                        raw = raw_components(data, K, Lhat) if L_ok else {"valid": False}
                        result, lam, valid = official(data, K, variant)

                        official_ell = result.get("ell_err_hat", mp.nan)
                        official_id = result.get("id_fie_hat", mp.nan)
                        official_log_id = (
                            mp.log(mp.fabs(official_id))
                            if valid and official_id != 0 and mp.isfinite(official_id)
                            else mp.nan
                        )

                        rows.append({
                            "function": name,
                            "function_label": label_for(name),
                            "replicate": rep,
                            "depth": depth,
                            "K": K,
                            "window_role": window_role(K),
                            "window_start_distance": mp.fabs(data[0]-L),
                            "latest_distance": mp.fabs(data[-1]-L),
                            "fraction_local": fraction_local,
                            "variant": variant,
                            "official_valid": valid,
                            "official_lambda_hat": lam,
                            "official_ell_err": official_ell,
                            "official_id_fie": official_id,
                            "official_log_abs_id_fie": official_log_id,
                            "manual_L_hat": Lhat,
                            "manual_L_abs_error": mp.fabs(Lhat-L) if L_ok else mp.nan,
                            "raw_valid": raw.get("valid", False),
                            "raw_lambda_hat": raw.get("lambda_hat", mp.nan),
                            "oracle_valid": oracle.get("valid", False),
                            "oracle_lambda_hat": oracle.get("lambda_hat", mp.nan),
                            "abs_official_minus_raw":
                                mp.fabs(lam-raw["lambda_hat"])
                                if valid and raw.get("valid", False) else mp.nan,
                        })

            print(f"Completed {name}, replicate={rep:02d}", flush=True)

    final_summary = summarize(rows)
    trend_summary = trends(rows, args.depths)

    raw_diffs = [
        r["abs_official_minus_raw"] for r in rows
        if r["official_valid"] and r["raw_valid"]
    ]
    implementation_checks = [{
        "check": "max_abs_official_minus_independent_raw_formula",
        "value": max(raw_diffs) if raw_diffs else mp.nan,
        "interpretation":
            "Zero indicates no clipping/floor/cap was detected relative to the independently recomputed raw formula.",
    }]

    write_csv(args.output/"detail.csv", rows)
    write_csv(args.output/"final_boundary_summary.csv", final_summary)
    write_csv(args.output/"final_boundary_trend_summary.csv", trend_summary)
    write_csv(args.output/"implementation_check_summary.csv", implementation_checks)

    config = {
        "experiment": "Experiment 4C final simple-root Newton boundary",
        "lambda_inf": "-infinity",
        "functions": {
            "poly_simple": "(x-2)(x^2+1), L=2",
            "exp_simple": "exp(x-1)-1, L=1",
        },
        "windows": args.windows,
        "window_roles": {
            "2,3,5": "local asymptotic diagnostics",
            "10": "transition",
            "20,40,50": "nonlocal-window controls",
        },
        "depths": args.depths,
        "cutoff": str(args.cutoff),
        "replicates": args.replicates,
        "seed": args.seed,
        "dps": args.dps,
        "primary_claim":
            "For genuinely local windows, the proposed estimate becomes increasingly negative as the Newton trajectory approaches a simple root with U'(L)=0.",
        "oracle_L": "diagnostic only",
        "independent_raw_recomputation": "diagnostic only; no clipping/floors/caps",
        "baselines": "not rerun here; this is a proposed-estimator boundary/scope experiment",
    }
    (args.output/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
