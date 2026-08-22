from src.systems import (
    generate_linear_trajectory,
    linear_true_lyapunov,
)

from src.estimators.proposed import (
    proposed_le_estimator,
    rolling_proposed_le_estimator,
)


def main():

    a = 0.8
    L = 0.0
    x0 = 1.0

    N = 30
    K = 10

    x = generate_linear_trajectory(
        a=a,
        L=L,
        x0=x0,
        N=N,
    )

    lambda_true = linear_true_lyapunov(a)

    # ==================================================
    # One specific rolling estimate
    # ==================================================

    n = 20

    result = proposed_le_estimator(
        x=x,
        K=K,
        end_index=n,
    )

    print()
    print("=" * 70)
    print("Single rolling-window test")
    print("=" * 70)

    print(f"K                  = {K}")
    print(f"end index n        = {result['end_index']}")
    print(f"boundary           = x_{result['boundary_index']}")
    print(
        f"window             = "
        f"x_{result['window_start']} ... "
        f"x_{result['window_end']}"
    )
    print(f"L_hat              = x_{result['L_hat_index']}")
    print()

    print(f"True LE            = {lambda_true:.12f}")
    print(f"Estimated LE       = {result['lambda_hat']:.12f}")
    print(f"ell_err_hat        = {result['ell_err_hat']:.12f}")
    print(f"ID_FIE_hat         = {result['id_fie_hat']:.12f}")
    print(f"L_hat value        = {result['L_hat']:.12e}")
    print(f"Valid              = {result['valid']}")

    print()

    # ==================================================
    # Every rolling estimate
    # ==================================================

    rolling = rolling_proposed_le_estimator(
        x=x,
        K=K,
    )

    print("=" * 70)
    print("All rolling estimates")
    print("=" * 70)

    print(
        f"{'n':>5} "
        f"{'window':>15} "
        f"{'L_hat index':>12} "
        f"{'LE estimate':>15}"
    )

    print("-" * 55)

    for result in rolling:

        print(
            f"{result['end_index']:5d} "
            f"{result['window_start']:4d}-"
            f"{result['window_end']:<4d} "
            f"{result['L_hat_index']:12d} "
            f"{result['lambda_hat']:15.10f}"
        )

    print("=" * 70)
    print()


if __name__ == "__main__":
    main()