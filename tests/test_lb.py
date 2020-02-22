from vlapy.core import lenard_bernstein
import numpy as np


def test_lenard_bernstein_maxwellian_solution():
    """
    tests if df/dt = 0 if f = maxwellian

    :return:
    """
    vmax = 6.0
    nv = 256
    dv = 2 * vmax / nv
    v = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)

    nu = 1e-3
    dt = 0.1
    v0 = 1.0

    leftside = lenard_bernstein.make_philharmonic_matrix(v, nv, nu, dt, dv, v0)

    f = np.exp(-(v ** 2.0) / 2.0 / v0) / np.sum(np.exp(-(v ** 2.0) / 2.0 / v0) * dv)
    f_out = f.copy()

    for it in range(32):
        f_out = lenard_bernstein.take_collision_step(leftside, f_out)

    np.testing.assert_almost_equal(f, f_out, decimal=4)


def test_lenard_bernstein_energy_conservation():
    """
    tests if the 2nd moment of f is conserved

    :return:
    """
    vmax = 6.0
    nv = 512
    dv = 2 * vmax / nv
    v = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)

    nu = 1e-3
    dt = 0.01
    v0 = 1.0

    leftside = lenard_bernstein.make_philharmonic_matrix(v, nv, nu, dt, dv, v0)

    f = np.exp(-((v - 0.5) ** 2.0) / 2.0 / v0)
    f = f / np.sum(f * dv)

    f_out = f.copy()
    for it in range(32):
        f_out = lenard_bernstein.take_collision_step(leftside, f_out)

    temp_in = np.sum(f * v ** 2.0) * dv
    temp_out = np.sum(f_out * v ** 2.0) * dv
    np.testing.assert_almost_equal(temp_out, temp_in, decimal=3)


def test_lenard_bernstein_density_conservation():
    """
    tests if the 0th moment of f is conserved

    :return:
    """
    vmax = 6.0
    nv = 512
    dv = 2 * vmax / nv
    v = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)

    nu = 1e-3
    dt = 0.1
    v0 = 1.0

    leftside = lenard_bernstein.make_philharmonic_matrix(v, nv, nu, dt, dv, v0)

    f = np.exp(-((v - 0.5) ** 2.0) / 2.0 / v0)
    f = f / np.sum(f * dv)

    f_out = f.copy()
    for it in range(32):
        f_out = lenard_bernstein.take_collision_step(leftside, f_out)

    temp_in = np.sum(f) * dv
    temp_out = np.sum(f_out) * dv
    np.testing.assert_almost_equal(temp_out, temp_in, decimal=3)


def test_lenard_bernstein_velocity_zero():
    """
    tests if the 1st moment of f is (approximately) 0

    :return:
    """
    vmax = 6.0
    nv = 256
    dv = 2 * vmax / nv
    v = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)

    nu = 5e-2
    dt = 0.1
    v0 = 1.0

    leftside = lenard_bernstein.make_philharmonic_matrix(v, nv, nu, dt, dv, v0)

    f = np.exp(-((v - 0.25) ** 2.0) / 2.0 / v0)
    f = f / np.sum(f * dv)

    f_out = f.copy()
    for it in range(500):
        f_out = lenard_bernstein.take_collision_step(leftside, f_out)

    temp_in = np.sum(f * v) * dv
    temp_out = np.sum(f_out * v) * dv
    np.testing.assert_almost_equal(temp_out, 0., decimal=1)
