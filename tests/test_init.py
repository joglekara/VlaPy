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
