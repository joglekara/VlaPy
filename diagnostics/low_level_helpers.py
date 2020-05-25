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


def get_first_mode(efield_arr):
    """
    Filters the electric field for only the first Fourier mode and reconstructs it back to real space

    :param efield_arr:
    :return:
    """
    ek = np.fft.fft(efield_arr.data, axis=1, norm="ortho")
    ek_rec = np.zeros(efield_arr.shape, dtype=np.complex)
    ek_rec[:, 1] = ek[:, 1]
    ek_rec = np.fft.ifft(ek_rec, axis=1, norm="ortho")

    return ek_rec


def get_w_ax(efield_arr):
    """
    Just gets the frequency axis

    :param efield_arr:
    :return:
    """
    tax = efield_arr.coords["time"].data
    dt = tax[1] - tax[0]
    wax = np.fft.fftfreq(tax.size, d=dt) * 2 * np.pi
    return wax


def get_e_max(efield_arr):
    """
    Gets E_max

    :param efield_arr:
    :return:
    """
    return np.amax(np.abs(efield_arr.data))


def get_damping_rate(efield_arr):
    """
    Gets the gradient of the last 75% of the simulation.

    TODO AJ - remove the hardcoding here

    :param efield_arr:
    :return:
    """
    tax = efield_arr.coords["time"].data

    t_ind = tax.size // 4

    ek = np.fft.fft(efield_arr.data, axis=1)
    ek_mag = np.array([np.abs(ek[it, 1]) for it in range(tax.size)])[t_ind:]

    dEdt = np.gradient(np.log(ek_mag), tax[2] - tax[1])

    return np.mean(dEdt)


def get_oscillation_frequency(efield_arr):
    tax = efield_arr.coords["time"].data

    # TODO AJ - need to come up with a better way to specify this index
    t_ind = tax.size // 4
    ekw = np.fft.fft2(efield_arr.data[t_ind:,])
    ek1w = np.abs(ekw[:, 1])

    wax = get_w_ax(efield_arr)

    return wax[ek1w.argmax()]
