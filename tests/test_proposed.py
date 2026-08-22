import numpy as np

from src.systems import (
    generate_linear_trajectory,
    linear_true_lyapunov,
)

from src.estimators.proposed import (
    proposed_le_estimator,
)


def main():

    # ----------------------------------------------
    # Simple clean linear contraction
    # ----------------------------------------------
    a = 0.8
    L = 1.0
    x0 = 3.0

    N = 30
    K = 10

    x = generate_linear_trajectory(
        a=a,
        L=L,
        x0=x0,
        N=N,
    )

    lambda_true = linear_true_lyapunov(a)

    result = proposed_le_estimator(
        x=x,
        K=K,
    )

    print()
    print("========================================")
    print("Proposed estimator smoke test")
    print("========================================")
    print(f"a                  = {a}")
    print(f"L                  = {L}")
    print(f"x0                 = {x0}")
    print(f"N                  = {N}")
    print(f"K                  = {K}")
    print(f"t0                 = {result['t0']}")
    print()

    print(f"True LE            = {lambda_true:.12f}")
    print(f"Estimated LE       = {result['lambda_hat']:.12f}")

    if result["valid"]:
        absolute_error = abs(
            result["lambda_hat"] - lambda_true
        )

        print(f"Absolute error     = {absolute_error:.12e}")

    print()
    print(f"ell_err_hat        = {result['ell_err_hat']:.12f}")
    print(f"ID_FIE_hat         = {result['id_fie_hat']:.12f}")
    print(f"L_hat              = {result['L_hat']:.12f}")
    print(f"Runtime (sec)      = {result['runtime_sec']:.8e}")
    print(f"Valid              = {result['valid']}")

    if not result["valid"]:
        print(f"Failure reason     = {result['failure_reason']}")

    print("========================================")
    print()


if __name__ == "__main__":
    main()