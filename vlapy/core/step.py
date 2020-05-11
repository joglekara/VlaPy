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

from vlapy.core import field, vlasov


def initialize(nx, nv, vmax=6.0):
    """
    Initializes a Maxwell-Boltzmann distribution

    TODO: temperature and density pertubations

    :param nx: size of grid in x (single int)
    :param nv: size of grid in v (single int)
    :param vmax: maximum absolute value of v (single float)
    :return:
    """

    f = np.zeros([nx, nv], dtype=np.float16)
    dv = 2.0 * vmax / nv
    vax = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)

    for ix in range(nx):
        f[ix,] = np.exp(-(vax ** 2.0) / 2.0)

    # normalize
    f = f / np.sum(f[0,]) / dv

    return f


def full_leapfrog_ps_step(f, x, kx, one_over_kx, v, kv, dv, t, dt, e, driver_function):
    """
    Takes a step forward in time for f and e

    Uses leapfrog scheme
    1 - spatial advection for 0.5 dt

    2a - field solve
    2b - velocity advection for dt

    3 - spatial advection for 0.5 dt

    :param f: distribution function. (numpy array of shape (nx, nv))
    :param x: real-space axis (numpy array of shape (nx,))
    :param kx: real-space wavenumber axis (numpy array of shape (nx,))
    :param v: velocity axis (numpy array of shape (nv,))
    :param kv: velocity-space wavenumber axis (numpy array of shape (nv,))
    :param dv: velocity-axis spacing (single float value)
    :param t: current time (single float value)
    :param dt: timestep (single float value)
    :param e: electric field (numpy array of shape (nx,))
    :param driver_function: function that returns an electric field (numpy array of shape (nx,))
    :return:
    """
    # f = vlasov.update_velocity_adv_spectral(f, kv, e, 0.5 * dt)
    # f = vlasov.update_spatial_adv_spectral(f, kx, v, dt)
    # e = field.get_total_electric_field(
    #     driver_function(x, t + dt), f=f, dv=dv, one_over_kx=one_over_kx
    # )
    # f = vlasov.update_velocity_adv_spectral(f, kv, e, 0.5 * dt)

    # f = vlasov.update_velocity_adv_sl(f, x, v, e, 0.5 * dt)
    f = vlasov.update_spatial_adv_sl(f, x, v, 0.5 * dt)
    e = field.get_total_electric_field(
        driver_function(x, t + 0.5 * dt), f=f, dv=dv, one_over_kx=one_over_kx
    )
    f = vlasov.update_velocity_adv_sl(f, x, v, e, dt)
    f = vlasov.update_spatial_adv_sl(f, x, v, 0.5 * dt)
    e = field.get_total_electric_field(
        driver_function(x, t + 0.5 * dt), f=f, dv=dv, one_over_kx=one_over_kx
    )

    return e, f


def full_PEFRL_ps_step(f, x, kx, one_over_kx, v, kv, dv, t, dt, e, driver_function):
    """
    Takes a step forward in time for f and e using the
    Performance-Extended Forest-Ruth-Like algorithm

    This is a 4th order symplectic integrator.
    http://physics.ucsc.edu/~peter/242/leapfrog.pdf

    :param f: distribution function. (numpy array of shape (nx, nv))

    :param x: real-space axis (numpy array of shape (nx,))
    :param kx: real-space wavenumber axis (numpy array of shape (nx,))

    :param v: velocity axis (numpy array of shape (nv,))
    :param kv: velocity-space wavenumber axis (numpy array of shape (nv,))
    :param dv: velocity-axis spacing (single float value)

    :param t: current time (single float value)
    :param dt: timestep (single float value)

    :param e: electric field (numpy array of shape (nv,))

    :param driver_function:
    :return:
    """
    xsi = 0.1786178958448091
    lambd = -0.2123418310626054
    chi = -0.6626458266981849e-1

    dt1 = xsi * dt
    dt2 = chi * dt
    dt3 = (1.0 - 2.0 * (chi + xsi)) * dt
    dt4 = dt2
    dt5 = dt1

    # x1
    f = vlasov.update_spatial_adv_spectral(f, kx, v, dt1)
    e = field.get_total_electric_field(
        driver_function(x, t + dt1), f=f, dv=dv, one_over_kx=one_over_kx
    )

    # v1
    f = vlasov.update_velocity_adv_spectral(f, kv, e, 0.5 * (1.0 - 2.0 * lambd) * dt)

    # x2
    f = vlasov.update_spatial_adv_spectral(f, kx, v, dt2)
    e = field.get_total_electric_field(
        driver_function(x, t + dt1 + dt2), f=f, dv=dv, one_over_kx=one_over_kx
    )

    # v2
    f = vlasov.update_velocity_adv_spectral(f, kv, e, lambd * dt)

    # x3
    f = vlasov.update_spatial_adv_spectral(f, kx, v, dt3)
    e = field.get_total_electric_field(
        driver_function(x, t + dt1 + dt2 + dt3), f=f, dv=dv, one_over_kx=one_over_kx
    )

    # v3
    f = vlasov.update_velocity_adv_spectral(f, kv, e, lambd * dt)

    # x4
    f = vlasov.update_spatial_adv_spectral(f, kx, v, dt4)
    e = field.get_total_electric_field(
        driver_function(x, t + dt1 + dt2 + dt3 + dt4),
        f=f,
        dv=dv,
        one_over_kx=one_over_kx,
    )

    # v4
    f = vlasov.update_velocity_adv_spectral(f, kv, e, 0.5 * (1.0 - 2.0 * lambd) * dt)

    # x5
    f = vlasov.update_spatial_adv_spectral(f, kx, v, dt5)
    e = field.get_total_electric_field(
        driver_function(x, t + dt1 + dt2 + dt3 + dt4 + dt5),
        f=f,
        dv=dv,
        one_over_kx=one_over_kx,
    )

    return e, f
