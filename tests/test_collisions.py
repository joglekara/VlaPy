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
from copy import deepcopy

from tests import helpers
from vlapy.core import step
import numpy as np


# ALL_SOLVERS = ["naive", "batched_tridiagonal"]
ALL_SOLVERS = ["batched_tridiagonal"]
ALL_OPERATORS = ["lb", "dg"]

TOLERANCE = 4
T_END = 16


def __initialize_for_collisions__(vshift):
    nx = 2
    nv = 1024

    nu = 1e-2
    dt = 0.1
    v0 = 1.0

    vmax = 6.0
    dv = 2 * vmax / nv
    v = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)

    f = helpers.__initialize_f__(nx=nx, v=v, v0=v0, vshift=vshift)

    return f, v, nv, nx, nu, dt, dv


def __run_collision_operator_test_loop__(
    vshift=0.0, collision_operator="lb", t_end=T_END, solver="naive"
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

    all_params = {
        "fokker-planck": {"type": collision_operator, "solver": solver},
        "nu": nu,
    }

    fp_step = step.get_collision_step(
        all_params=all_params, stuff_for_time_loop=stuff_for_time_loop
    )

    f_out = deepcopy(f)
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
        vshift=0.0, t_end=T_END, collision_operator=collision_operator, solver=solver
    )

    np.testing.assert_almost_equal(f, f_out, decimal=4)


def test_energy_conservation():
    for collision_operator, solver in product(ALL_OPERATORS, ALL_SOLVERS):
        __test_energy_conservation__(collision_operator, solver)


def __test_energy_conservation__(collision_operator, solver):
    """
    tests if the 2nd moment of f is conserved

    :return:
    """
    f, f_out, v, dv = __run_collision_operator_test_loop__(
        vshift=0.5, t_end=T_END, collision_operator=collision_operator, solver=solver
    )

    temp_in = np.trapz(f * v[None, :] ** 2.0, dx=dv, axis=1)
    temp_out = np.trapz(f_out * v[None, :] ** 2.0, dx=dv, axis=1)
    np.testing.assert_almost_equal(temp_out, temp_in, decimal=4)


def test_density_conservation():
    for collision_operator, solver in product(ALL_OPERATORS, ALL_SOLVERS):
        __test_density_conservation__(collision_operator, solver)


def __test_density_conservation__(collision_operator, solver):
    """
    tests if the 0th moment of f is conserved

    :return:
    """

    f, f_out, v, dv = __run_collision_operator_test_loop__(
        vshift=0.5, t_end=T_END, collision_operator=collision_operator, solver=solver
    )

    temp_in = np.trapz(f, dx=dv, axis=1)
    temp_out = np.trapz(f_out, dx=dv, axis=1)
    np.testing.assert_almost_equal(temp_out, temp_in, decimal=TOLERANCE)


def test_momentum_conservation_if_initialized_at_zero():
    for collision_operator, solver in product(ALL_OPERATORS, ALL_SOLVERS):
        __test_momentum_conservation_if_initialized_at_zero__(
            collision_operator=collision_operator, solver=solver
        )


def __test_momentum_conservation_if_initialized_at_zero__(collision_operator, solver):
    """
    tests if the 1st moment of f is conserved if initialized at 0

    :return:
    """
    f, f_out, v, dv = __run_collision_operator_test_loop__(
        vshift=0.0, t_end=T_END, collision_operator=collision_operator, solver=solver
    )

    temp_in = np.trapz(f * v[None, :], dx=dv, axis=1)
    temp_out = np.trapz(f_out * v[None, :], dx=dv, axis=1)

    np.testing.assert_almost_equal(actual=temp_in, desired=temp_out, decimal=TOLERANCE)


def test_momentum_conservation():
    for collision_operator, solver in product(ALL_OPERATORS, ALL_SOLVERS):
        __test_momentum_conservation__(
            collision_operator=collision_operator, solver=solver
        )


def __test_momentum_conservation__(collision_operator, solver):
    """
    Tests if the 1st moment of f is conserved or not

    This moment is not conserved and is brought to 0 by the following operators:
    1 - "lb"

    This moment is conserved by the following operators:
    1 - "dg"


    :return:
    """
    f, f_out, v, dv = __run_collision_operator_test_loop__(
        vshift=1.5, collision_operator=collision_operator, solver=solver, t_end=T_END,
    )

    temp_in = np.trapz(f * v[None, :], dx=dv, axis=1)
    temp_out = np.trapz(f_out * v[None, :], dx=dv, axis=1)

    if collision_operator == "lb":
        np.testing.assert_array_less(x=temp_out, y=temp_in)
    elif collision_operator == "dg":
        np.testing.assert_almost_equal(
            actual=temp_in, desired=temp_out, decimal=TOLERANCE
        )
