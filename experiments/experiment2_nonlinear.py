import csv
from pathlib import Path

import numpy as np
import mpmath as mp

from src.systems import (
    generate_nonlinear_trajectory,
    nonlinear_true_lyapunov,
    nonlinear_finite_window_lyapunov,
)

from src.estimators.proposed import (
    rolling_proposed_le_estimator,
)


# ============================================================
# Experiment parameters
# ============================================================

N = 200

L = 2.0


A_VALUES = [
    -0.95,
    -0.80,
    -0.60,
    -0.40,
    -0.20,
     0.20,
     0.40,
     0.60,
     0.80,
     0.95,
]


K_VALUES = [
    10,
    20,
    40,
    50,
]


LIMIT_METHODS = [
    "next",
    "aitken",
]


# ============================================================
# Nonlinearity strength
# ============================================================
#
# We parameterize beta through eta:
#
#     |beta|
#     =
#     eta * (1 - |a|) / (2d)
#
# where
#
#     d = |x0 - L|.
#
# Therefore, on the initial interval
#
#     |x-L| <= d,
#
# we have
#
#     |U'(x)|
#     <=
#     |a| + 2|beta|d
#
#     =
#     |a| + eta(1-|a|)
#
#     < 1
#
# whenever eta < 1.
#
# eta = 0:
#     affine sanity case
#
# eta = 0.5:
#     moderate nonlinear contribution
#
# eta = 0.9:
#     strong nonlinear contribution
#     while retaining the contraction bound.
# ============================================================

ETA_VALUES = [
    0.0,
    0.4,
    0.9,
]


# ============================================================
# Numerical precision
# ============================================================

MP_DPS = 200

mp.mp.dps = MP_DPS


# ============================================================
# Reproducibly random initial condition
# ============================================================

RANDOM_SEED = 2026

rng = np.random.default_rng(
    RANDOM_SEED
)

X0 = float(
    rng.uniform(
        L - 1.0,
        L + 1.0,
    )
)

while X0 == L:

    X0 = float(
        rng.uniform(
            L - 1.0,
            L + 1.0,
        )
    )


# ============================================================
# Helpers
# ============================================================

def mp_value(value):
    """
    Convert without passing through binary float
    whenever possible.
    """

    if isinstance(value, mp.mpf):
        return value

    return mp.mpf(
        str(value)
    )


def mp_string(
    value,
    digits=50,
):
    """
    String representation for CSV output.
    """

    if value == mp.inf:
        return "inf"

    if value == mp.ninf:
        return "-inf"

    if not mp.isfinite(value):
        return "nan"

    return mp.nstr(
        value,
        digits,
    )


def beta_cases_for_a(
    a,
    L,
    x0,
):
    """
    Construct beta values from eta.

    For eta = 0:
        only beta = 0.

    For eta > 0:
        test both +beta and -beta.
    """

    a_mp = mp_value(a)
    L_mp = mp_value(L)
    x0_mp = mp_value(x0)

    d = mp.fabs(
        x0_mp - L_mp
    )

    if d == 0:
        raise ValueError(
            "x0 must differ from L."
        )

    cases = []

    for eta in ETA_VALUES:

        eta_mp = mp_value(
            eta
        )

        if eta == 0.0:

            cases.append({
                "eta":
                    eta_mp,

                "beta_sign":
                    0,

                "beta":
                    mp.mpf("0"),

                "q_bound":
                    mp.fabs(a_mp),
            })

            continue

        beta_magnitude = (
            eta_mp
            * (
                1
                - mp.fabs(a_mp)
            )
            / (
                2 * d
            )
        )

        for beta_sign in [
            -1,
            1,
        ]:

            beta = (
                mp.mpf(beta_sign)
                * beta_magnitude
            )

            q_bound = (
                mp.fabs(a_mp)
                + 2
                * mp.fabs(beta)
                * d
            )

            # This should hold by construction.
            if not q_bound < 1:

                raise RuntimeError(
                    "Constructed nonlinear map "
                    "does not satisfy q_bound < 1."
                )

            cases.append({
                "eta":
                    eta_mp,

                "beta_sign":
                    beta_sign,

                "beta":
                    beta,

                "q_bound":
                    q_bound,
            })

    return cases


# ============================================================
# Main experiment
# ============================================================

def run_experiment():

    output_dir = Path(
        "results/tables"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rolling_file = (
        output_dir
        / "experiment2_nonlinear_rolling.csv"
    )

    summary_file = (
        output_dir
        / "experiment2_nonlinear_summary.csv"
    )

    rolling_rows = []
    summary_rows = []

    L_mp = mp_value(
        L
    )

    # ========================================================
    # Header
    # ========================================================

    print()
    print("=" * 190)

    print(
        "Experiment 2: Nonlinear "
        "Fixed-Point Lyapunov Estimation"
    )

    print("=" * 190)

    print(
        "U(x) = "
        "L + a(x-L) + beta(x-L)^2"
    )

    print()

    print(
        f"N           = {N}"
    )

    print(
        f"L           = {L}"
    )

    print(
        f"x0          = {X0:.16f}"
    )

    print(
        f"random seed = {RANDOM_SEED}"
    )

    print(
        f"K           = {K_VALUES}"
    )

    print(
        f"eta         = {ETA_VALUES}"
    )

    print(
        f"precision   = {MP_DPS} decimal digits"
    )

    print(
        f"methods     = {LIMIT_METHODS}"
    )

    print()

    header = (
        f"{'a':>7} "
        f"{'eta':>6} "
        f"{'sgn':>4} "
        f"{'beta':>12} "
        f"{'K':>4} "
        f"{'method':>8} "
        f"{'valid':>10} "
        f"{'lambda_inf':>13} "
        f"{'lambda_FT':>13} "
        f"{'lambda_hat':>13} "
        f"{'ell_err':>13} "
        f"{'FIE':>10} "
        f"{'|Lhat-L|':>12} "
        f"{'err_inf':>12} "
        f"{'err_FT':>12} "
        f"{'status':>10}"
    )

    print(header)
    print("-" * len(header))

    # ========================================================
    # a
    # ========================================================

    for a in A_VALUES:

        lambda_inf = (
            nonlinear_true_lyapunov(
                a=a,
                high_precision=True,
            )
        )

        beta_cases = (
            beta_cases_for_a(
                a=a,
                L=L,
                x0=X0,
            )
        )

        # ====================================================
        # beta
        # ====================================================

        for beta_case in beta_cases:

            eta = (
                beta_case["eta"]
            )

            beta_sign = (
                beta_case[
                    "beta_sign"
                ]
            )

            beta = (
                beta_case["beta"]
            )

            q_bound = (
                beta_case["q_bound"]
            )

            # ------------------------------------------------
            # Generate one trajectory for this (a, beta)
            # ------------------------------------------------

            x = (
                generate_nonlinear_trajectory(
                    a=a,
                    beta=beta,
                    L=L,
                    x0=X0,
                    N=N,
                    high_precision=True,
                )
            )

            # =================================================
            # K
            # =================================================

            for K in K_VALUES:

                # =================================================
                # limit method
                # =================================================

                for limit_method in LIMIT_METHODS:

                    results = (
                        rolling_proposed_le_estimator(
                            x=x,
                            K=K,
                            limit_method=limit_method,
                            numeric_backend="mpmath",
                        )
                    )

                    total_windows = (
                        len(results)
                    )

                    valid_windows = sum(
                        result["valid"]
                        for result in results
                    )

                    # =============================================
                    # Store every rolling window
                    # =============================================

                    for result in results:

                        n = (
                            result[
                                "end_index"
                            ]
                        )

                        lambda_ft = (
                            nonlinear_finite_window_lyapunov(
                                x=x,
                                a=a,
                                beta=beta,
                                L=L,
                                K=K,
                                end_index=n,
                                high_precision=True,
                            )
                        )

                        if result["valid"]:

                            L_hat_error = (
                                mp.fabs(
                                    result["L_hat"]
                                    - L_mp
                                )
                            )

                            error_inf = (
                                mp.fabs(
                                    result[
                                        "lambda_hat"
                                    ]
                                    - lambda_inf
                                )
                            )

                            if mp.isfinite(
                                lambda_ft
                            ):

                                error_ft = (
                                    mp.fabs(
                                        result[
                                            "lambda_hat"
                                        ]
                                        - lambda_ft
                                    )
                                )

                            else:

                                error_ft = (
                                    mp.nan
                                )

                        else:

                            L_hat_error = (
                                mp.nan
                            )

                            error_inf = (
                                mp.nan
                            )

                            error_ft = (
                                mp.nan
                            )

                        rolling_rows.append({

                            "a":
                                a,

                            "eta":
                                mp_string(
                                    eta
                                ),

                            "beta_sign":
                                beta_sign,

                            "beta":
                                mp_string(
                                    beta
                                ),

                            "q_bound":
                                mp_string(
                                    q_bound
                                ),

                            "K":
                                K,

                            "N":
                                N,

                            "L":
                                L,

                            "x0":
                                X0,

                            "random_seed":
                                RANDOM_SEED,

                            "precision_dps":
                                MP_DPS,

                            "limit_method":
                                limit_method,

                            "end_index":
                                n,

                            "window_start":
                                result[
                                    "window_start"
                                ],

                            "window_end":
                                result[
                                    "window_end"
                                ],

                            "latest_observation_index":
                                result[
                                    "latest_observation_index"
                                ],

                            "lambda_inf_true":
                                mp_string(
                                    lambda_inf
                                ),

                            "lambda_ft_true":
                                mp_string(
                                    lambda_ft
                                ),

                            "lambda_hat":
                                mp_string(
                                    result[
                                        "lambda_hat"
                                    ]
                                ),

                            "ell_err_hat":
                                mp_string(
                                    result[
                                        "ell_err_hat"
                                    ]
                                ),

                            "id_fie_hat":
                                mp_string(
                                    result[
                                        "id_fie_hat"
                                    ]
                                ),

                            "L_hat":
                                mp_string(
                                    result[
                                        "L_hat"
                                    ]
                                ),

                            "L_hat_error":
                                mp_string(
                                    L_hat_error
                                ),

                            "absolute_error_to_lambda_inf":
                                mp_string(
                                    error_inf
                                ),

                            "absolute_error_to_lambda_ft":
                                mp_string(
                                    error_ft
                                ),

                            "runtime_sec":
                                result[
                                    "runtime_sec"
                                ],

                            "valid":
                                result[
                                    "valid"
                                ],

                            "failure_reason":
                                result[
                                    "failure_reason"
                                ],
                        })

                    # =============================================
                    # FINAL WINDOW ONLY
                    #
                    # Do NOT replace a failed final Aitken
                    # calculation with an earlier successful one.
                    #
                    # If it fails at n=N, report the failure.
                    # =============================================

                    final_result = (
                        results[-1]
                    )

                    final_n = (
                        final_result[
                            "end_index"
                        ]
                    )

                    final_lambda_ft = (
                        nonlinear_finite_window_lyapunov(
                            x=x,
                            a=a,
                            beta=beta,
                            L=L,
                            K=K,
                            end_index=final_n,
                            high_precision=True,
                        )
                    )

                    # ---------------------------------------------
                    # Valid final estimate
                    # ---------------------------------------------

                    if final_result["valid"]:

                        final_L_error = (
                            mp.fabs(
                                final_result[
                                    "L_hat"
                                ]
                                - L_mp
                            )
                        )

                        final_error_inf = (
                            mp.fabs(
                                final_result[
                                    "lambda_hat"
                                ]
                                - lambda_inf
                            )
                        )

                        if mp.isfinite(
                            final_lambda_ft
                        ):

                            final_error_ft = (
                                mp.fabs(
                                    final_result[
                                        "lambda_hat"
                                    ]
                                    - final_lambda_ft
                                )
                            )

                        else:

                            final_error_ft = (
                                mp.nan
                            )

                        status = "OK"

                        print(
                            f"{a:7.2f} "
                            f"{float(eta):6.2f} "
                            f"{beta_sign:4d} "
                            f"{float(beta):12.4e} "
                            f"{K:4d} "
                            f"{limit_method:>8} "
                            f"{valid_windows:4d}/{total_windows:<5d} "
                            f"{float(lambda_inf):13.7f} "
                            f"{float(final_lambda_ft):13.7f} "
                            f"{float(final_result['lambda_hat']):13.7f} "
                            f"{float(final_result['ell_err_hat']):13.7f} "
                            f"{float(final_result['id_fie_hat']):10.6f} "
                            f"{float(final_L_error):12.3e} "
                            f"{float(final_error_inf):12.3e} "
                            f"{float(final_error_ft):12.3e} "
                            f"{status:>10}"
                        )

                    # ---------------------------------------------
                    # Invalid final estimate
                    # ---------------------------------------------

                    else:

                        final_L_error = (
                            mp.nan
                        )

                        final_error_inf = (
                            mp.nan
                        )

                        final_error_ft = (
                            mp.nan
                        )

                        status = "INVALID"

                        print(
                            f"{a:7.2f} "
                            f"{float(eta):6.2f} "
                            f"{beta_sign:4d} "
                            f"{float(beta):12.4e} "
                            f"{K:4d} "
                            f"{limit_method:>8} "
                            f"{valid_windows:4d}/{total_windows:<5d} "
                            f"{float(lambda_inf):13.7f} "
                            f"{float(final_lambda_ft):13.7f} "
                            f"{'nan':>13} "
                            f"{'nan':>13} "
                            f"{'nan':>10} "
                            f"{'nan':>12} "
                            f"{'nan':>12} "
                            f"{'nan':>12} "
                            f"{status:>10}"
                        )

                    # =============================================
                    # Summary row
                    # =============================================

                    summary_rows.append({

                        "a":
                            a,

                        "eta":
                            mp_string(
                                eta
                            ),

                        "beta_sign":
                            beta_sign,

                        "beta":
                            mp_string(
                                beta
                            ),

                        "q_bound":
                            mp_string(
                                q_bound
                            ),

                        "K":
                            K,

                        "N":
                            N,

                        "L":
                            L,

                        "x0":
                            X0,

                        "random_seed":
                            RANDOM_SEED,

                        "precision_dps":
                            MP_DPS,

                        "limit_method":
                            limit_method,

                        "end_index":
                            final_n,

                        "valid_windows":
                            valid_windows,

                        "total_windows":
                            total_windows,

                        "lambda_inf_true":
                            mp_string(
                                lambda_inf
                            ),

                        "lambda_ft_true":
                            mp_string(
                                final_lambda_ft
                            ),

                        "lambda_hat":
                            mp_string(
                                final_result[
                                    "lambda_hat"
                                ]
                            ),

                        "ell_err_hat":
                            mp_string(
                                final_result[
                                    "ell_err_hat"
                                ]
                            ),

                        "id_fie_hat":
                            mp_string(
                                final_result[
                                    "id_fie_hat"
                                ]
                            ),

                        "L_hat":
                            mp_string(
                                final_result[
                                    "L_hat"
                                ]
                            ),

                        "L_hat_error":
                            mp_string(
                                final_L_error
                            ),

                        "absolute_error_to_lambda_inf":
                            mp_string(
                                final_error_inf
                            ),

                        "absolute_error_to_lambda_ft":
                            mp_string(
                                final_error_ft
                            ),

                        "runtime_sec":
                            final_result[
                                "runtime_sec"
                            ],

                        "valid":
                            final_result[
                                "valid"
                            ],

                        "status":
                            status,

                        "failure_reason":
                            final_result[
                                "failure_reason"
                            ],
                    })

    # ========================================================
    # Save rolling CSV
    # ========================================================

    if rolling_rows:

        with open(
            rolling_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=rolling_rows[0].keys(),
            )

            writer.writeheader()

            writer.writerows(
                rolling_rows
            )

    # ========================================================
    # Save summary CSV
    # ========================================================

    if summary_rows:

        with open(
            summary_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=summary_rows[0].keys(),
            )

            writer.writeheader()

            writer.writerows(
                summary_rows
            )

    # ========================================================
    # Finish
    # ========================================================

    print()
    print("=" * 190)

    print(
        "Summary saved to:"
    )

    print(
        f"  {summary_file}"
    )

    print()

    print(
        "All rolling estimates saved to:"
    )

    print(
        f"  {rolling_file}"
    )

    print()

    print(
        f"Total rolling estimates: "
        f"{len(rolling_rows)}"
    )

    print("=" * 190)
    print()


if __name__ == "__main__":
    run_experiment()