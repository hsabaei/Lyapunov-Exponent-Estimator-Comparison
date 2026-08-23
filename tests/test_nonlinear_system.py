import numpy as np

from src.systems import (
    nonlinear_map,
    nonlinear_map_derivative,
)


def test_nonlinear_fixed_point():

    L = 2.0
    a = 0.8
    beta = 0.1

    value = nonlinear_map(
        x=L,
        a=a,
        beta=beta,
        L=L,
    )

    assert np.isclose(
        value,
        L,
    )


def test_nonlinear_derivative_at_limit():

    L = 2.0
    a = -0.6
    beta = 0.2

    derivative = (
        nonlinear_map_derivative(
            x=L,
            a=a,
            beta=beta,
            L=L,
        )
    )

    assert np.isclose(
        derivative,
        a,
    )


def test_asymptotic_le_is_log_abs_a():

    a = -0.6

    expected = np.log(
        abs(a)
    )

    assert np.isclose(
        expected,
        -0.5108256237659907,
    )