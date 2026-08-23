import numpy as np

from src.estimators.yang_wu import (
    yang_wu_scalar_le_estimator,
)


def affine_trajectory(a, L, x0, N):
    x = np.empty(N + 1, dtype=float)
    x[0] = x0

    for t in range(N):
        x[t + 1] = L + a * (x[t] - L)

    return x


def nonlinear_trajectory(
    a,
    beta,
    L,
    x0,
    N,
):
    x = np.empty(N + 1, dtype=float)
    x[0] = x0

    for t in range(N):
        e = x[t] - L
        x[t + 1] = (
            L
            + a * e
            + beta * e**2
        )

    return x


def main():

    L = 2.0
    x0 = 1.3578696273508724

    print("\n=== AFFINE VALIDATION ===")

    for a in [
        -0.95,
        -0.8,
        -0.6,
        -0.4,
        -0.2,
        0.2,
        0.4,
        0.6,
        0.8,
        0.95,
    ]:

        x = affine_trajectory(
            a=a,
            L=L,
            x0=x0,
            N=200,
        )

        result = (
            yang_wu_scalar_le_estimator(
                x=x,
                evolution=1,
                r_min=None,
            )
        )

        truth = np.log(abs(a))

        if result["valid"]:
            error = abs(
                result["lambda_hat"]
                - truth
            )
        else:
            error = np.nan

        n_unique = np.unique(x).size

        print(
            f"a={a:+.2f} "
            f"truth={truth:+.8f} "
            f"YW={result['lambda_hat']:+.8f} "
            f"error={error:.3e} "
            f"valid_maps="
            f"{result['n_valid_local_maps']}/"
            f"{result['n_attempted_local_maps']} "
            f"unique_states={n_unique}/{len(x)} "
            f"valid={result['valid']}"
        )

    print(
        "\n=== NONLINEAR VALIDATION ==="
    )

    # Representative Experiment-2 case.
    a = 0.8
    eta = 0.9

    d = abs(x0 - L)

    beta = (
        eta
        * (1.0 - abs(a))
        / (2.0 * d)
    )

    x = nonlinear_trajectory(
        a=a,
        beta=beta,
        L=L,
        x0=x0,
        N=200,
    )

    result = (
        yang_wu_scalar_le_estimator(
            x=x,
            evolution=1,
            r_min=None,
        )
    )

    truth = np.log(abs(a))

    print(
        f"a={a:+.2f} "
        f"eta={eta:.1f} "
        f"beta={beta:+.8f}"
    )

    print(
        f"asymptotic truth = {truth:+.8f}"
    )

    print(
        f"Yang-Wu estimate = "
        f"{result['lambda_hat']:+.8f}"
    )

    print(
        f"absolute error = "
        f"{abs(result['lambda_hat'] - truth):.3e}"
    )

    print(
        f"valid local maps = "
        f"{result['n_valid_local_maps']}/"
        f"{result['n_attempted_local_maps']}"
    )

    print(
        f"unique states = "
        f"{np.unique(x).size}/{len(x)}"
    )

    print(
        f"valid = {result['valid']}"
    )


if __name__ == "__main__":
    main()