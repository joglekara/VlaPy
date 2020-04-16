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

import numpy as np


def make_philharmonic_matrix(vax, nv, nu, dt, dv, v0):
    """
    This matrix is composed of the linear operator that must be inverted with respect
    to the right side, which'll be the distribution function

    :param vax:
    :param nv:
    :param nu:
    :param dt:
    :param dv:
    :param v0:
    :return:
    """

    a = nu * dt * np.ones(nv - 1) * (-(v0 ** 2.0) / dv ** 2.0 + vax[:-1] / 2 / dv)
    b = nu * dt * np.ones(nv) * (2 * v0 ** 2.0 / dv ** 2.0 + 0.0 / 2 / dv)
    c = nu * dt * np.ones(nv - 1) * (-(v0 ** 2.0) / dv ** 2.0 - vax[1:] / 2 / dv)
    leftside = np.diag(a, -1) + np.diag(1 + b, 0) + np.diag(c, 1)

    return leftside


def take_collision_step(leftside, f):
    """
    Just solves a tridiagonal system here.

    :param leftside:
    :param f_t:
    :return:
    """
    return np.linalg.solve(leftside, f)
