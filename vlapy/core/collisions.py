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
    """
    This function returns the function for preparing the matrix representing the Lenard-Bernstein [1] collision operator.

    It uses the arguments to create static arrays for the linear operato.

    [1] : Lenard, A., & Bernstein, I. B. (1958). Plasma oscillations with diffusion in velocity space.
        Physical Review, 112(5), 1456–1459. https://doi.org/10.1103/PhysRev.112.1456

    :param vax: (1D float array) - 1D array representing the centers of the velocity grid
    :param nv: (int) the size of the velocity grid
    :param nx: (int) the size of the spatial grid
    :param nu: (float) the collision frequency
    :param dt: (float) the timestep
    :param dv: (float) the velocity grid spacing
    :return: new function with above arguments initialized as static variables
    """

    def make_philharmonic_arrays_for_matrix(f_xv):
        """
        This function creates the arrays representing the Lenard-Bernstein [1] collision operator. When
        center-differenced to second order in velocity space, this operator can be represented by a tridiagonal matrix.
        The result of this function is used in a tridiagonal matrix solver.

        [1] : Lenard, A., & Bernstein, I. B. (1958). Plasma oscillations with diffusion in velocity space.
        Physical Review, 112(5), 1456–1459. https://doi.org/10.1103/PhysRev.112.1456

        :param f_xv: The distribution function used to calculate the quantities in the linear operator.
        :return: a, b, c -- The subdiagonal, diagonal, and super-diagonal, respectively representing the LB operator.
        """

        v0t_sq = np.trapz(
            f_xv
            * vax[
                None,
            ]
            ** 2.0,
            dx=dv,
            axis=1,
        )

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
    """
    This function returns the function for preparing the matrix representing the Dougherty [1] collision operator.

    It uses the arguments to create static arrays for the linear operato.

    [1] : Dougherty, J. P. (1964). Model Fokker-Planck Equation for a Plasma and Its Solution.
    Physics of Fluids, 7(11), 1788. https://doi.org/10.1063/1.2746779

    :param vax: (1D float array) - 1D array representing the centers of the velocity grid
    :param nv: (int) the size of the velocity grid
    :param nx: (int) the size of the spatial grid
    :param nu: (float) the collision frequency
    :param dt: (float) the timestep
    :param dv: (float) the velocity grid spacing
    :return: new function with above arguments initialized as static variables
    """

    def make_dougherty_arrays_for_matrix(f_xv):
        """
            This function creates the arrays representing the Dougherty [1] collision operator. When
            center-differenced to second order in velocity space, this operator can be represented by a tridiagonal matrix.
            The result of this function is used in a tridiagonal matrix solver.

            [1] : Dougherty, J. P. (1964). Model Fokker-Planck Equation for a Plasma and Its Solution.
        Physics of Fluids, 7(11), 1788. https://doi.org/10.1063/1.2746779

            :param f_xv: The distribution function used to calculate the quantities in the linear operator.
            :return: a, b, c -- The subdiagonal, diagonal, and super-diagonal, respectively representing the LB operator.
        """

        vbar = np.trapz(
            f_xv
            * vax[
                None,
            ],
            dx=dv,
            axis=1,
        )
        v0t_sq = np.trapz(
            f_xv
            * (
                vax[
                    None,
                ]
                - vbar[:, None]
            )
            ** 2.0,
            dx=dv,
            axis=1,
        )

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
    """
    This function returns the naive solver for the collision operator. Specifically, this is simply a for loop
    over the all the x-indices where each f(v) is solved for.

    :param nx: (int) number of x cells
    :return: new function with above arguments initialized as static variables
    """

    def _solver_(a, b, c, f):
        """
        For each value in x, this solves an `Af_p = b` problem where `A` is formed using the arguments
         `a,b,c`. The solution, `f_p`, is obtained using numpy's matrix solver.

        :param a: (2D float array (nx, nv-1)) the sub-diagonal of the matrix for each x cell
        :param b: (2D float array (nx, nv)) the diagonal of the matrix
        :param c: (2D float array (nx, nv-1)) the super-diagonal of the matrix
        :param f: the RHS of the `Af_p = f` system.
        :return: the solution, `x`, of the `Af_p = b` system
        """
        for ix in range(nx):
            leftside = (
                np.diag(
                    np.squeeze(
                        a[
                            ix,
                        ]
                    ),
                    -1,
                )
                + np.diag(
                    np.squeeze(
                        b[
                            ix,
                        ]
                    ),
                    0,
                )
                + np.diag(
                    np.squeeze(
                        c[
                            ix,
                        ]
                    ),
                    1,
                )
            )
            f[ix,] = np.linalg.solve(
                leftside,
                f[
                    ix,
                ],
            )

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
