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
from scipy import fft

def compute_charges(f, dv):
    """
    Computes a simple moment of the distribution function along
    the velocity axis using the trapezoidal rule

    :param f: distribution function. (numpy array of shape (nx, nv))
    :param dv: velocity-axis spacing (float)
    :return:
    """
    return np.trapz(f, dx=dv, axis=1)


def __fft_solve__(net_charge_density, one_over_kx):
    """
    del^2 phi = -rho
    del e = - integral[rho] = - integral[fdv]

    :param net_charge_density: charge-density (numpy array of shape (nx,))
    :param one_over_kx: one over real-space wavenumber axis (numpy array of shape (nx,))
    :return:
    """

    return np.real(fft.ifft(1j * one_over_kx * fft.fft(net_charge_density)))


def solve_for_field(charge_density, one_over_kx):
    """
    Solves for the net electric field after subtracting ion charge

    :param charge_density: charge-density (numpy array of shape (nx,))
    :param one_over_kx: one over real-space wavenumber axis (numpy array of shape (nx,))

    :return:
    """
    return __fft_solve__(
        net_charge_density=1.0 - charge_density, one_over_kx=one_over_kx
    )


def get_spectral_solver(dv, one_over_kx):
    """
    This function gets the field solver

    :param dv:
    :param one_over_kx:
    :return:
    """

    def solve_total_electric_field(driver_field, f):
        """
        Allows adding a driver field

        :param driver_field: an electric field (numpy array of shape (nx,))
        :param f: distribution function. (numpy array of shape (nx, nv))
        :param dv: velocity-axis spacing (float)
        :param one_over_kx: one over real-space wavenumber axis (numpy array of shape (nx,))
        :return:
        """
        return driver_field + solve_for_field(
            charge_density=compute_charges(f, dv), one_over_kx=one_over_kx
        )

    return solve_total_electric_field


def get_field_solver(stuff_for_time_loop, field_solver_implementation="spectral"):

    if field_solver_implementation == "spectral":
        field_solver = get_spectral_solver(
            dv=stuff_for_time_loop["dv"], one_over_kx=stuff_for_time_loop["one_over_kx"]
        )
    else:
        raise NotImplementedError

    return field_solver
