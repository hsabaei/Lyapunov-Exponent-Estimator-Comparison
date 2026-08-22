import numpy as np

from src.systems import (
    generate_linear_trajectory,
    linear_true_lyapunov,
)

from src.estimators.proposed import (
    proposed_le_estimator,
)


# ============================================================
# Diagnostic parameters
# ============================================================

N = 60
L = 0.0
X0 = 1.0

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

K_VALUES = [10, 15, 20, 40]


def main():

    print()
    print("=" * 118)
    print("Oracle fixed-point diagnostic")
    print("=" * 118)

    header = (
        f"{'a':>7} "
        f"{'K':>5} "
        f"{'True LE':>14} "
        f"{'Practical LE':>14} "
        f"{'Oracle LE':>14} "
        f"{'Oracle err':>13} "
        f"{'Oracle ell_err':>16} "
        f"{'Oracle FIE':>12}"
    )

    print(header)
    print("-" * len(header))

    all_pass = True

    for a in A_VALUES:

        x = generate_linear_trajectory(
            a=a,
            L=L,
            x0=X0,
            N=N,
        )

        lambda_true = linear_true_lyapunov(a)

        for K in K_VALUES:

            # Any rolling position should give the same
            # result for the exactly linear system.
            #
            # Use the first available endpoint.
            n = K + 1

            # --------------------------------------------
            # Normal practical estimator
            # L_hat = x_{n+1}
            # --------------------------------------------

            practical = proposed_le_estimator(
                x=x,
                K=K,
                end_index=n,
            )

            # --------------------------------------------
            # Oracle diagnostic
            # L_hat = true L
            # --------------------------------------------

            oracle = proposed_le_estimator(
                x=x,
                K=K,
                end_index=n,
                L_hat_override=L,
            )

            if oracle["valid"]:

                oracle_error = abs(
                    oracle["lambda_hat"]
                    - lambda_true
                )

                passed = np.isclose(
                    oracle["lambda_hat"],
                    lambda_true,
                    rtol=1e-10,
                    atol=1e-12,
                )

            else:
                oracle_error = np.nan
                passed = False

            all_pass = all_pass and passed

            print(
                f"{a:7.2f} "
                f"{K:5d} "
                f"{lambda_true:14.8f} "
                f"{practical['lambda_hat']:14.8f} "
                f"{oracle['lambda_hat']:14.8f} "
                f"{oracle_error:13.3e} "
                f"{oracle['ell_err_hat']:16.8f} "
                f"{oracle['id_fie_hat']:12.8f}"
            )

    print()
    print("=" * 118)

    if all_pass:
        print("PASS: All oracle-limit estimates recover the exact linear Lyapunov exponent.")
    else:
        print("FAIL: At least one oracle-limit estimate does not recover the exact exponent.")

    print("=" * 118)
    print()


if __name__ == "__main__":
    main()