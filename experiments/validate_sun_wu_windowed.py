from mpmath import mp

from src.estimators.sun_wu import (
    sun_wu_scalar_rbf_le_estimator,
)


MP_DPS = 200
N = 200
L = mp.mpf("2.0")
X0 = mp.mpf("1.3578696273508724")

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


def affine_trajectory(a):
    a = mp.mpf(a)
    x = [X0]

    for _ in range(N + 1):
        x.append(
            L + a * (x[-1] - L)
        )

    return x


def nonlinear_trajectory(a, beta):
    a = mp.mpf(a)
    beta = mp.mpf(beta)
    x = [X0]

    for _ in range(N + 1):
        e = x[-1] - L
        x.append(
            L + a * e + beta * e * e
        )

    return x


def final_window(x, K):
    # Same observed record used in the current final-window comparison:
    # x_{N-K}, ..., x_N, x_{N+1}.
    return x[N - K : N + 2]


def main():
    mp.dps = MP_DPS

    print("\n=== SUN-WU WINDOWED AFFINE VALIDATION ===")

    for a_text in A_VALUES:
        a = mp.mpf(a_text)
        truth = mp.log(mp.fabs(a))
        x = affine_trajectory(a)

        for K in K_VALUES:
            result = sun_wu_scalar_rbf_le_estimator(
                final_window(x, K)
            )

            if result["valid"]:
                error = abs(
                    float(result["lambda_hat"])
                    - float(truth)
                )
            else:
                error = float("nan")

            print(
                f"a={float(a):+.2f} "
                f"K={K:2d} "
                f"truth={float(truth):+.8f} "
                f"SW={float(result['lambda_hat']):+.8f} "
                f"error={error:.3e} "
                f"hidden={result['selected_hidden_nodes']} "
                f"pairs={result['n_pairs']} "
                f"train_mse={result['training_mse']:.3e} "
                f"valid={result['valid']}"
            )

    print("\n=== SUN-WU WINDOWED NONLINEAR VALIDATION ===")

    a = mp.mpf("0.8")
    eta = mp.mpf("0.9")

    d = mp.fabs(X0 - L)

    beta = (
        eta
        * (1 - mp.fabs(a))
        / (2 * d)
    )

    x = nonlinear_trajectory(
        a=a,
        beta=beta,
    )

    truth = mp.log(mp.fabs(a))

    print(
        f"a={float(a):+.2f} "
        f"eta={float(eta):.1f} "
        f"beta={float(beta):+.8f}"
    )

    for K in K_VALUES:
        result = sun_wu_scalar_rbf_le_estimator(
            final_window(x, K)
        )

        if result["valid"]:
            error = abs(
                float(result["lambda_hat"])
                - float(truth)
            )
        else:
            error = float("nan")

        print(
            f"K={K:2d} "
            f"truth={float(truth):+.8f} "
            f"SW={float(result['lambda_hat']):+.8f} "
            f"error={error:.3e} "
            f"hidden={result['selected_hidden_nodes']} "
            f"pairs={result['n_pairs']} "
            f"train_mse={result['training_mse']:.3e} "
            f"valid={result['valid']}"
        )


if __name__ == "__main__":
    main()
