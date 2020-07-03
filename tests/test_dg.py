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

from vlapy.core import collisions
import numpy as np


def test_dg_maxwellian_solution():
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

    f = np.exp(-(v ** 2.0))
    f = f / np.sum(f * dv)
    f_out = f.copy()

    for it in range(8):
        f_out = collisions.take_collision_step(
            collisions.make_daugherty_matrix, f_out, v, nv, nu, dt, dv
        )

    np.testing.assert_almost_equal(f, f_out, decimal=5)


def test_dg_energy_conservation():
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

    f = np.exp(-((v - 0.5) ** 2.0))
    f = f / np.sum(f * dv)
    f_out = f.copy()

    for it in range(8):
        f_out = collisions.take_collision_step(
            collisions.make_daugherty_matrix, f_out, v, nv, nu, dt, dv
        )

    temp_in = np.sum(f * v ** 2.0) * dv
    temp_out = np.sum(f_out * v ** 2.0) * dv
    np.testing.assert_almost_equal(temp_out, temp_in, decimal=6)


def test_dg_density_conservation():
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

    f = np.exp(-((v - 0.5) ** 2.0))
    f = f / np.sum(f * dv)
    f_out = f.copy()

    for it in range(8):
        f_out = collisions.take_collision_step(
            collisions.make_daugherty_matrix, f_out, v, nv, nu, dt, dv
        )

    temp_in = np.sum(f) * dv
    temp_out = np.sum(f_out) * dv
    np.testing.assert_almost_equal(temp_out, temp_in, decimal=6)


# def test_dg_velocity_zero():
#     """
#     tests if the 1st moment of f is (approximately) 0
#
#     :return:
#     """
#     vmax = 6.0
#     nv = 256
#     dv = 2 * vmax / nv
#     v = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)
#
#     nu = 5e-2
#     dt = 0.1
#
#     f = np.exp(-((v - 0.1) ** 2.0))
#     f = f / np.sum(f * dv)
#     f_out = f.copy()
#
#     for it in range(50000):
#         f_out = collisions.take_collision_step(
#             collisions.make_daugherty_matrix, f_out, v, nv, nu, dt, dv
#         )
#
#     temp_in = np.sum(f * v) * dv
#     temp_out = np.sum(f_out * v) * dv
#     np.testing.assert_almost_equal(temp_out, 0.0, decimal=4)
