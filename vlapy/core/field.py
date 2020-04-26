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


def compute_charges(f, dv):
    """
    Computes a simple moment of the distribution function along
    the velocity axis using the trapezoidal rule

    :param f:
    :param dv:
    :return:
    """
    return np.trapz(f, dx=dv, axis=1)


def __fft_solve__(net_charge_density, kx):
    """
    del^2 phi = -rho
    del e = - integral[rho] = - integral[fdv]

    :param net_charge_density:
    :param kx:
    :return:
    """
    rhok = np.fft.fft(net_charge_density)
    rhok[0] = 0
    rhok[1:] = rhok[1:] / (-1j * kx[1:])

    return np.real(np.fft.ifft(rhok))


def solve_for_field(charge_density, kx):
    """
    Solves for the net electric field after subtracting ion charge

    :param charge_density:
    :param kx:

    :return:
    """
    return __fft_solve__(net_charge_density=1.0 - charge_density, kx=kx)


def get_total_electric_field(driver_field, f, dv, kx):
    """
    Allows adding a driver field

    :param driver_field:
    :param f:
    :param dv:
    :param kx:
    :return:
    """
    return driver_field + solve_for_field(charge_density=compute_charges(f, dv), kx=kx)
