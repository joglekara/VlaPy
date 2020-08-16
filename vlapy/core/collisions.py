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


def get_philharmonic_matrix_maker(vax, nv, nx, nu, dt, dv):
    def make_philharmonic_arrays_for_matrix(f_xv):
        """
        These are for use with the vectorized collision operator

        :param v: velocity axis (numpy array of shape (nv,))
        :param nv: size of velocity axis (int)
        :param nu: collision frequency (float)
        :param dt: timestep (float)
        :param dv: velocity-axis spacing (float)
        :param v0: thermal temperature (float)
        :return leftside: matrix for the linear operator
        """

        v0t_sq = np.trapz(f_xv * vax[None,] ** 2.0, dx=dv, axis=1)

        a = (
            nu
            * dt
            * np.ones((nx, nv - 1))
            * (-v0t_sq[:, None] / dv ** 2.0 + vax[None, :-1] / 2 / dv)
        )
        b = 1.0 + nu * dt * np.ones((nx, nv)) * (2 * v0t_sq[:, None] / dv ** 2.0)
        c = (
            nu
            * dt
            * np.ones((nx, nv - 1))
            * (-v0t_sq[:, None] / dv ** 2.0 - vax[None, 1:] / 2 / dv)
        )

        return a, b, c

    return make_philharmonic_arrays_for_matrix


def get_dougherty_matrix_maker(vax, nv, nx, nu, dt, dv):
    def make_dougherty_arrays_for_matrix(f_xv):
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

        vbar = np.trapz(f_xv * vax[None,], dx=dv, axis=1)
        v0t_sq = np.trapz(f_xv * (vax[None,] - vbar[:, None]) ** 2.0, dx=dv, axis=1)

        a = (
            nu
            * dt
            * np.ones((nx, nv - 1))
            * (
                -v0t_sq[:, None] / dv ** 2.0
                + (vax[None, :-1] - vbar[:, None]) / 2.0 / dv
            )
        )
        b = 1.0 + nu * dt * np.ones((nx, nv)) * (2.0 * v0t_sq[:, None] / dv ** 2.0)
        c = (
            nu
            * dt
            * np.ones((nx, nv - 1))
            * (
                -v0t_sq[:, None] / dv ** 2.0
                - (vax[None, 1:] - vbar[:, None]) / 2.0 / dv
            )
        )

        return a, b, c

    return make_dougherty_arrays_for_matrix


def get_naive_solver(nx):
    def _solver_(a, b, c, f):
        """
        returns the solution using numpy's solver

        :param leftside:
        :param f:
        :return:
        """
        for ix in range(nx):
            leftside = (
                np.diag(np.squeeze(a[ix,]), -1)
                + np.diag(np.squeeze(b[ix,]), 0)
                + np.diag(np.squeeze(c[ix,]), 1)
            )
            f[ix,] = np.linalg.solve(leftside, f[ix,])

        return f

    return _solver_


def get_batched_tridiag_solver(nv):
    def _batched_tridiag_solver_(a, b, c, f):
        """
        Arrayed/Sliced algorithm for tridiagonal solve.

        About 50x faster than looping through a numpy linalg call
        for a 256x2048 solve

        :param a:
        :param b:
        :param c:
        :param f:
        :param nv:
        :return:
        """

        ac = a.copy()
        bc = b.copy()
        cc = c.copy()
        dc = f.copy()

        for it in range(1, nv):
            mc = ac[:, it - 1] / bc[:, it - 1]
            bc[:, it] = bc[:, it] - mc * cc[:, it - 1]
            dc[:, it] = dc[:, it] - mc * dc[:, it - 1]

        xc = bc
        xc[:, -1] = dc[:, -1] / bc[:, -1]

        for il in range(nv - 2, -1, -1):
            xc[:, il] = (dc[:, il] - cc[:, il] * xc[:, il + 1]) / bc[:, il]

        return xc

    return _batched_tridiag_solver_


def get_matrix_solver(nx, nv, solver_name="batched_tridiagonal"):

    if solver_name == "naive":
        matrix_solver = get_naive_solver(nx)
    elif solver_name == "batched_tridiagonal":
        matrix_solver = get_batched_tridiag_solver(nv)
    else:
        raise NotImplementedError

    return matrix_solver


def get_batched_array_maker(vax, nv, nx, nu, dt, dv, operator="lb"):

    if operator == "lb":
        get_matrix_maker = get_philharmonic_matrix_maker
    elif operator == "dg":
        get_matrix_maker = get_dougherty_matrix_maker
    else:
        raise NotImplementedError

    return get_matrix_maker(vax, nv, nx, nu, dt, dv)
