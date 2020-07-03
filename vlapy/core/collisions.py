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


def make_philharmonic_matrix(v, nv, nu, dt, dv, f_v):
    """
    This matrix is composed of the linear operator that must be inverted with respect
    to the right side, which'll be the distribution function

    :param v: velocity axis (numpy array of shape (nv,))
    :param nv: size of velocity axis (int)
    :param nu: collision frequency (float)
    :param dt: timestep (float)
    :param dv: velocity-axis spacing (float)
    :param v0: thermal temperature (float)
    :return leftside: matrix for the linear operator
    """
    v0t_sq = np.trapz(f_v * v ** 2.0, dx=dv, axis=0)

    a = nu * dt * np.ones(nv - 1) * (-v0t_sq / dv ** 2.0 + v[:-1] / 2 / dv)
    b = nu * dt * np.ones(nv) * (2 * v0t_sq / dv ** 2.0 + 0.0 / 2 / dv)
    c = nu * dt * np.ones(nv - 1) * (-v0t_sq / dv ** 2.0 - v[1:] / 2 / dv)
    leftside = np.diag(a, -1) + np.diag(1 + b, 0) + np.diag(c, 1)

    return leftside


def make_daugherty_matrix(v, nv, nu, dt, dv, f_v):
    """
    This matrix is composed of the linear operator that must be inverted with respect
    to the right side, which'll be the distribution function

    :param v: velocity axis (numpy array of shape (nv,))
    :param nv: size of velocity axis (int)
    :param nu: collision frequency (float)
    :param dt: timestep (float)
    :param dv: velocity-axis spacing (float)
    :param v0: thermal temperature (float)
    :return leftside: matrix for the linear operator
    """
    vbar = np.trapz(f_v * v, dx=dv, axis=0)
    v0t_sq = np.trapz(f_v * (v - vbar) ** 2.0, dx=dv, axis=0)

    a = nu * dt * np.ones(nv - 1) * (-v0t_sq / dv ** 2.0 + (v[:-1] - vbar) / 2.0 / dv)
    b = nu * dt * np.ones(nv) * (2.0 * v0t_sq / dv ** 2.0)
    c = nu * dt * np.ones(nv - 1) * (-v0t_sq / dv ** 2.0 - (v[1:] - vbar) / 2.0 / dv)
    leftside = np.diag(a, -1) + np.diag(1 + b, 0) + np.diag(c, 1)

    return leftside


def take_collision_step(operator_method, f, v, nv, nu, dt, dv):
    """
    Just solves a tridiagonal system here.

    :param leftside: matrix containing system of equations (numpy array of shape (nv, nv))
    :param f: distribution function at a single point in space. (numpy array of shape (nv, ))
    :return solution to leftside x = f:
    """
    return np.linalg.solve(operator_method(f_v=f, v=v, nv=nv, nu=nu, dt=dt, dv=dv), f)
