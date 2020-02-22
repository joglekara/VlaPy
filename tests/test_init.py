from vlapy.core import step
import numpy as np


def test_initial_density():
    nx = 64
    nv = 1024
    vmax = 6.0
    dv = 2 * vmax / nv
    f = step.initialize(nx, nv)

    np.testing.assert_almost_equal(f[0,].sum() * dv, np.ones(nx), decimal=3)


def test_initial_temperature():
    nx = 64
    nv = 1024
    vmax = 6.0
    dv = 2 * vmax / nv
    v = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)
    f = step.initialize(nx, nv)

    np.testing.assert_almost_equal(
        np.array(
            [
                (
                    (f[ix, 1:-1] * v[1:-1] ** 2.0).sum()
                    + 0.5 * (f[ix, 0] * v[0] + f[ix, -1]) * v[-1]
                )
                * dv
                for ix in range(nx)
            ]
        ),
        np.ones(nx),
        decimal=3,
    )
