# MIT License
#
# Copyright (c) 2020 Archis Joglekar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from itertools import product

from tests import helpers
from vlapy.core import step
import numpy as np


ALL_SOLVERS = ["naive", "batched_tridiagonal"]
ALL_OPERATORS = ["lb", "dg"]


def __initialize_for_collisions__(vshift):
    nx = 1
    nv = 256

    nu = 1e-3
    dt = 0.1
    v0 = 1.0

    vmax = 6.0
    dv = 2 * vmax / nv
    v = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)

    f = helpers.__initialize_f__(nx=nx, v=v, v0=v0, vshift=vshift)

    return f, v, nv, nx, nu, dt, dv


def __run_collision_operator_test_loop__(
    vshift=0.0, collision_operator="lb", t_end=32, solver="naive"
):
    f, v, nv, nx, nu, dt, dv = __initialize_for_collisions__(vshift=vshift)

    stuff_for_time_loop = {
        "f": f,
        "v": v,
        "nv": nv,
        "nx": nx,
        "nu": nu,
        "dt": dt,
        "dv": dv,
    }

    all_params = {"fokker-planck": {"type": collision_operator, "solver": solver}}

    fp_step = step.get_collision_step(
        all_params=all_params, stuff_for_time_loop=stuff_for_time_loop
    )

    f_out = f.copy()
    for it in range(t_end):

        f_out = fp_step(f_out)

    return f, f_out, v, dv


def test_maxwellian_solution():
    for collision_operator, solver in product(ALL_OPERATORS, ALL_SOLVERS):
        __test_maxwellian_solution__(collision_operator, solver)


def __test_maxwellian_solution__(collision_operator, solver):
    """
    tests if df/dt = 0 if f = maxwellian

    :return:
    """

    f, f_out, v, dv = __run_collision_operator_test_loop__(
        vshift=0.0, t_end=32, collision_operator=collision_operator, solver=solver
    )

    np.testing.assert_almost_equal(f, f_out, decimal=6)


def test_energy_conservation():
    for collision_operator, solver in product(ALL_OPERATORS, ALL_SOLVERS):
        __test_energy_conservation__(collision_operator, solver)


def __test_energy_conservation__(collision_operator, solver):
    """
    tests if the 2nd moment of f is conserved

    :return:
    """
    f, f_out, v, dv = __run_collision_operator_test_loop__(
        vshift=0.5, t_end=32, collision_operator=collision_operator, solver=solver
    )

    temp_in = np.trapz(f[None,] * v ** 2.0, dx=dv, axis=1)
    temp_out = np.trapz(f_out[None,] * v ** 2.0, dx=dv, axis=1)
    np.testing.assert_almost_equal(temp_out, temp_in, decimal=6)


def test_density_conservation():
    for collision_operator, solver in product(ALL_OPERATORS, ALL_SOLVERS):
        __test_density_conservation__(collision_operator, solver)


def __test_density_conservation__(collision_operator, solver):
    """
    tests if the 0th moment of f is conserved

    :return:
    """

    f, f_out, v, dv = __run_collision_operator_test_loop__(
        vshift=0.5, t_end=32, collision_operator=collision_operator, solver=solver
    )

    temp_in = np.trapz(f[None,], dx=dv, axis=1)
    temp_out = np.trapz(f_out[None,], dx=dv, axis=1)
    np.testing.assert_almost_equal(temp_out, temp_in, decimal=6)


def test_lenard_bernstein_momentum_conservation_if_initialized_at_zero():
    for collision_operator, solver in product(ALL_OPERATORS, ALL_SOLVERS):
        __test_momentum_conservation_if_initialized_at_zero__(
            collision_operator=collision_operator, solver=solver
        )


def __test_momentum_conservation_if_initialized_at_zero__(collision_operator, solver):
    """
    tests if the 0th moment of f is conserved

    :return:
    """
    f, f_out, v, dv = __run_collision_operator_test_loop__(
        vshift=0.0, t_end=32, collision_operator=collision_operator, solver=solver
    )

    temp_in = np.trapz(f[None,] * v, dx=dv, axis=1)
    temp_out = np.trapz(f_out[None,] * v, dx=dv, axis=1)

    np.testing.assert_almost_equal(actual=temp_in, desired=temp_out, decimal=6)


def test_dougherty_momentum_conservation():
    for collision_operator, solver in product(ALL_OPERATORS, ALL_SOLVERS):
        __test_momentum_conservation__(
            collision_operator=collision_operator, solver=solver
        )


def __test_momentum_conservation__(collision_operator, solver):
    """
    tests if the 0th moment of f is conserved

    :return:
    """
    f, f_out, v, dv = __run_collision_operator_test_loop__(
        vshift=0.1, collision_operator=collision_operator, solver=solver
    )

    temp_in = np.trapz(f[None,] * v, dx=dv, axis=1)
    temp_out = np.trapz(f_out[None,] * v, dx=dv, axis=1)

    np.testing.assert_almost_equal(actual=temp_in, desired=temp_out, decimal=6)


# def test_lenard_bernstein_velocity_zero_np():
#     __test_lenard_bernstein_velocity_zero(solver="numpy")
#     pass
#
#
# def test_lenard_bernstein_velocity_zero_bt():
#     __test_lenard_bernstein_velocity_zero(solver="batched_tridiagonal")
#     pass
#
#
# def __test_lenard_bernstein_velocity_zero(solver, collision_operator):
#     """
#     tests if the 1st moment of f is (approximately) 0
#
#     :return:
#     """
#
#     f, f_out, v, dv = helpers.__run_collision_operator_test_loop(
#         vshift=0.1, solver=solver, t_end=1000, collision_operator=collision_operator,
#     )
#
#     temp_out = np.trapz(f_out[None,] * v, dx=dv, axis=1)
#
#     np.testing.assert_almost_equal(actual=0.0, desired=temp_out, decimal=4)
