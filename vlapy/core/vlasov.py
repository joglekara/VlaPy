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
from scipy import interpolate


def __get_k__(ax):
    """
    get axis of transformed quantity

    :param ax: axis to transform
    :return:
    """
    return np.fft.fftfreq(ax.size, d=ax[1] - ax[0])


def update_spatial_adv_sl(f, x, v, dt):
    """
    evolution of df/dt = v df/dx

    :param f: distribution function. (numpy array of shape (nx, nv))
    :param kx: real-space wavenumber axis (numpy array of shape (nx,))
    :param v: velocity axis (numpy array of shape (nv,))
    :param dt: timestep (single float value)
    :return:
    """

    x_pad = np.zeros(x.size + 2)
    x_pad[1:-1] = x
    x_pad[0] = x[0] - (x[2] - x[1])
    x_pad[-1] = x[-1] + (x[2] - x[1])

    f_pad = np.zeros((x_pad.size, v.size))
    f_pad[1:-1, :] = f
    f_pad[0, :] = f[-1, :]
    f_pad[-1, :] = f[0, :]

    xm, vm = np.meshgrid(x, v, indexing="ij")
    xm = xm.flatten()
    vm = vm.flatten()

    f_interpolator = interpolate.RectBivariateSpline(x_pad, v, f_pad)
    f_out = f_interpolator(xm - vm * dt, vm, grid=False).reshape((x.size, v.size))

    # f_out = interpolate.griddata(
    #     (xm, vm), f.flatten(), (xm - vm * dt, vm), method="cubic"
    # ).reshape((x.size, v.size))

    return f_out


def update_spatial_adv_spectral(f, kx, v, dt):
    """
    evolution of df/dt = v df/dx

    :param f: distribution function. (numpy array of shape (nx, nv))
    :param kx: real-space wavenumber axis (numpy array of shape (nx,))
    :param v: velocity axis (numpy array of shape (nv,))
    :param dt: timestep (single float value)
    :return:
    """

    return np.real(np.fft.ifft(__vdfdx__(np.fft.fft(f, axis=0), v, kx, dt), axis=0))


def update_velocity_adv_spectral(f, kv, e, dt):
    """
    evolution of df/dt = e df/dv

    :param f: distribution function. (numpy array of shape (nx, nv))
    :param kv: velocity-space wavenumber axis (numpy array of shape (nv,))
    :param e: electric field (numpy array of shape (nx,))
    :param dt: timestep (single float value)
    :return:
    """

    return np.real(np.fft.ifft(__edfdv__(np.fft.fft(f, axis=1), e, kv, dt), axis=1))


def update_velocity_adv_sl(f, x, v, e, dt):
    """
    evolution of df/dt = e df/dv

    :param f: distribution function. (numpy array of shape (nx, nv))
    :param kv: velocity-space wavenumber axis (numpy array of shape (nv,))
    :param e: electric field (numpy array of shape (nx,))
    :param dt: timestep (single float value)
    :return:
    """

    v_pad = np.zeros(v.size + 2)
    v_pad[1:-1] = v
    v_pad[0] = v[0] - (v[2] - v[1])
    v_pad[-1] = v[-1] + (v[2] - v[1])

    f_pad = np.zeros((x.size, v_pad.size))
    f_pad[:, 1:-1] = f
    f_pad[:, 0] = f[:, -1]
    f_pad[:, -1] = f[:, 0]

    e_fit = interpolate.interp1d(x, e, kind="cubic")

    xm, vm = np.meshgrid(x, v, indexing="ij")

    xm = xm.flatten()
    vm = vm.flatten()

    em = e_fit(xm)

    f_interpolator = interpolate.RectBivariateSpline(x, v_pad, f_pad)
    f_out = f_interpolator(xm, vm - em * dt, grid=False).reshape((x.size, v.size))

    # f_out = interpolate.griddata(
    #     (xm, vm), f.flatten(), (xm, vm - em * dt), method="cubic"
    # ).reshape((x.size, v.size))

    return f_out


def __edfdv__(f, e, kv, dt):
    """
    Lowest level routine for Edf/dv. This operator is in Fourier space


    :param f: distribution function. (numpy array of shape (nx, nv))
    :param e: electric field (numpy array of shape (nx,))
    :param kv: velocity-space wavenumber axis (numpy array of shape (nv,))
    :param dt: timestep (single float value)
    :return:
    """
    return np.exp(-1j * kv * dt * e[:, None]) * f


def __vdfdx__(f, v, kx, dt):
    """
    Lowest level routine for vdf/dx. This operator is in Fourier space

    :param f: distribution function. (numpy array of shape (nx, nv))
    :param v: velocity axis (numpy array of shape (nv,))
    :param kx: real-space wavenumber axis (numpy array of shape (nx,))
    :param dt: timestep (single float value)
    :return:
    """
    return np.exp(-1j * kx[:, None] * dt * v) * f
