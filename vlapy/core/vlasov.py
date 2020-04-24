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


def __get_k__(ax):
    """
    get axis of transformed quantity

    :param ax:
    :return:
    """
    return np.fft.fftfreq(ax.size, d=ax[1] - ax[0])


def update_spatial_adv_spectral(f, kx, v, dt):
    """
    evolution of df/dt = v df/dx

    :param f:
    :param kx:
    :param v:
    :param dt:
    :return:
    """

    return np.real(np.fft.ifft(__vdfdx__(np.fft.fft(f, axis=0), v, kx, dt), axis=0))


def update_velocity_adv_spectral(f, kv, e, dt):
    """
    evolution of df/dt = e df/dv

    :param f:
    :param kv:
    :param e:
    :param dt:
    :return:
    """

    return np.real(np.fft.ifft(__edfdv__(np.fft.fft(f, axis=1), e, kv, dt), axis=1))


def __edfdv__(fp, e, kv, dt):
    """

    :param fp:
    :param e:
    :param kv:
    :param dt:
    :return:
    """
    return np.exp(-1j * kv * dt * e[:, None]) * fp


def __vdfdx__(fp, v, kx, dt):
    """

    :param fp:
    :param v:
    :param kx:
    :param dt:
    :return:
    """
    return np.exp(-1j * kx[:, None] * dt * v) * fp
