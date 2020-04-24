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

from vlapy.core import field
import numpy as np


def test_field_solver():
    nx = 96
    kx_pert = 0.25
    xmax = 2 * np.pi / kx_pert
    dx = xmax / nx
    axis = np.linspace(dx / 2, xmax - dx / 2, nx)
    kx = np.fft.fftfreq(axis.size, d=dx) * 2.0 * np.pi

    charge_densities = [
        1.0 + np.sin(kx_pert * axis),
        1.0 + np.cos(2 * kx_pert * axis),
        1.0 + np.sin(2 * kx_pert * axis) + np.cos(8 * kx_pert * axis),
    ]

    electric_fields = [
        np.cos(kx_pert * axis) / kx_pert,
        -np.sin(2 * kx_pert * axis) / 2.0 / kx_pert,
        np.cos(2 * kx_pert * axis) / 2.0 / kx_pert
        - np.sin(8 * kx_pert * axis) / 8.0 / kx_pert,
    ]

    for actual_field, charge_density in zip(electric_fields, charge_densities):
        test_field = field.solve_for_field(charge_density=1.0 - charge_density, kx=kx)
        np.testing.assert_almost_equal(actual_field, test_field, decimal=4)
