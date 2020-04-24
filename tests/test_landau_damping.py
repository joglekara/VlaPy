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

from vlapy.core import step, field
from diagnostics.z_function import get_roots_to_electrostatic_dispersion


def test_full_leapfrog_ps_step_landau_damping():

    nx = 64
    nv = 512

    f = step.initialize(nx, nv)

    k0 = 0.3
    w_complex = get_roots_to_electrostatic_dispersion(1.0, 1.0, k0)
    w0 = np.real(w_complex)
    actual_decay_rate = np.imag(w_complex)

    xmax = 2 * np.pi / k0
    xmin = 0.0

    dx = (xmax - xmin) / nx
    x = np.linspace(xmin + dx / 2.0, xmax - dx / 2.0, nx)
    kx = np.fft.fftfreq(x.size, d=dx) * 2.0 * np.pi

    vmax = 6.0
    dv = 2 * vmax / nv
    v = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)
    kv = np.fft.fftfreq(v.size, d=dv) * 2.0 * np.pi

    nt = 1000
    tmax = 100
    t = np.linspace(0, tmax, nt)
    dt = t[1] - t[0]

    def driver_function(x, t):
        """
        t is 0D
        x is 1D

        """
        a0 = 4e-4
        envelope = np.exp(-((t - 8) ** 8.0) / 4.0 ** 8.0)

        return envelope * a0 * np.cos(k0 * x + w0 * t)

    field_store = np.zeros([nt, nx])
    dist_store = np.zeros([nt, nx, nv])
    t_store = np.zeros(nt)

    e = field.get_total_electric_field(driver_function(x, t[0]), f=f, dv=dv, kx=kx)

    it = 0
    t_store[it] = t[it]
    dist_store[it] = f
    field_store[it] = e

    for it in range(1, nt):
        e, f = step.full_leapfrog_ps_step(
            f, x, kx, v, kv, dv, t[it], dt, e, driver_function
        )

        t_store[it] = t[it]
        dist_store[it] = f
        field_store[it] = e

    t_ind = 600
    ek = np.fft.fft(field_store, axis=1)
    ek_mag = np.array([np.abs(ek[it, 1]) for it in range(nt)])
    decay_rate = np.mean(np.gradient(np.log(ek_mag[-t_ind:]), 0.1))

    np.testing.assert_almost_equal(decay_rate, actual_decay_rate, decimal=2)

    ekw = np.fft.fft2(field_store[nt // 2 :,])
    ek1w = np.abs(ekw[:, 1])
    wax = np.fft.fftfreq(ek1w.shape[0], d=dt) * 2 * np.pi

    np.testing.assert_almost_equal(wax[ek1w.argmax()], w0, decimal=1)


def test_full_leapfrog_ps_step_zero():

    nx = 32
    nv = 512

    f = step.initialize(nx, nv)

    # f - defined
    # w0 = 1.1056
    # k0 = 0.25

    w0 = 1.1598
    k0 = 0.3

    # w0 = 1.2850
    # k0 = 0.4

    xmax = 2 * np.pi / k0
    xmin = 0.0

    dx = (xmax - xmin) / nx
    x = np.linspace(xmin + dx / 2.0, xmax - dx / 2.0, nx)
    kx = np.fft.fftfreq(x.size, d=dx) * 2.0 * np.pi

    vmax = 6.0
    dv = 2 * vmax / nv
    v = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)
    kv = np.fft.fftfreq(v.size, d=dv) * 2.0 * np.pi

    nt = 400
    tmax = 40.0
    t = np.linspace(0, tmax, nt)
    dt = t[1] - t[0]

    def driver_function(x, t):
        """
        t is 0D
        x is 1D

        """
        a0 = 1e-6
        envelope = np.exp(-((t - 8) ** 8.0) / 4.0 ** 8.0)

        return 0 * envelope * a0 * np.cos(k0 * x + w0 * t)

    field_store = np.zeros([nt, nx])
    dist_store = np.zeros([nt, nx, nv])
    t_store = np.zeros(nt)

    e = field.get_total_electric_field(driver_function(x, t[0]), f=f, dv=dv, kx=kx)

    it = 0
    t_store[it] = t[it]
    dist_store[it] = f
    field_store[it] = e

    for it in range(1, nt):
        e, f = step.full_leapfrog_ps_step(
            f, x, kx, v, kv, dv, t[it], dt, e, driver_function
        )

        t_store[it] = t[it]
        dist_store[it] = f
        field_store[it] = e

    np.testing.assert_almost_equal(field_store[0], field_store[-1], decimal=2)
    np.testing.assert_almost_equal(dist_store[0], dist_store[-1], decimal=2)


def test_full_PEFRL_ps_step_landau_damping():

    nx = 64
    nv = 512

    f = step.initialize(nx, nv)

    k0 = 0.3
    w_complex = get_roots_to_electrostatic_dispersion(1.0, 1.0, k0)
    w0 = np.real(w_complex)
    actual_decay_rate = np.imag(w_complex)

    xmax = 2 * np.pi / k0
    xmin = 0.0

    dx = (xmax - xmin) / nx
    x = np.linspace(xmin + dx / 2.0, xmax - dx / 2.0, nx)
    kx = np.fft.fftfreq(x.size, d=dx) * 2.0 * np.pi

    vmax = 6.0
    dv = 2 * vmax / nv
    v = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)
    kv = np.fft.fftfreq(v.size, d=dv) * 2.0 * np.pi

    nt = 1000
    tmax = 100
    t = np.linspace(0, tmax, nt)
    dt = t[1] - t[0]

    def driver_function(x, t):
        """
        t is 0D
        x is 1D

        """
        a0 = 4e-4
        envelope = np.exp(-((t - 8) ** 8.0) / 4.0 ** 8.0)

        return envelope * a0 * np.cos(k0 * x + w0 * t)

    field_store = np.zeros([nt, nx])
    dist_store = np.zeros([nt, nx, nv])
    t_store = np.zeros(nt)

    e = field.get_total_electric_field(driver_function(x, t[0]), f=f, dv=dv, kx=kx)

    it = 0
    t_store[it] = t[it]
    dist_store[it] = f
    field_store[it] = e

    for it in range(1, nt):
        e, f = step.full_PEFRL_ps_step(
            f, x, kx, v, kv, dv, t[it], dt, e, driver_function
        )

        t_store[it] = t[it]
        dist_store[it] = f
        field_store[it] = e

    t_ind = 600
    ek = np.fft.fft(field_store, axis=1)
    ek_mag = np.array([np.abs(ek[it, 1]) for it in range(nt)])
    decay_rate = np.mean(np.gradient(np.log(ek_mag[-t_ind:]), 0.1))

    np.testing.assert_almost_equal(decay_rate, actual_decay_rate, decimal=2)

    ekw = np.fft.fft2(field_store[nt // 2 :,])
    ek1w = np.abs(ekw[:, 1])
    wax = np.fft.fftfreq(ek1w.shape[0], d=dt) * 2 * np.pi

    np.testing.assert_almost_equal(wax[ek1w.argmax()], w0, decimal=1)


def test_full_pefrl_ps_step_zero():

    nx = 32
    nv = 512

    f = step.initialize(nx, nv)

    # f - defined
    # w0 = 1.1056
    # k0 = 0.25

    w0 = 1.1598
    k0 = 0.3

    # w0 = 1.2850
    # k0 = 0.4

    xmax = 2 * np.pi / k0
    xmin = 0.0

    dx = (xmax - xmin) / nx
    x = np.linspace(xmin + dx / 2.0, xmax - dx / 2.0, nx)
    kx = np.fft.fftfreq(x.size, d=dx) * 2.0 * np.pi

    vmax = 6.0
    dv = 2 * vmax / nv
    v = np.linspace(-vmax + dv / 2.0, vmax - dv / 2.0, nv)
    kv = np.fft.fftfreq(v.size, d=dv) * 2.0 * np.pi

    nt = 400
    tmax = 40.0
    t = np.linspace(0, tmax, nt)
    dt = t[1] - t[0]

    def driver_function(x, t):
        """
        t is 0D
        x is 1D

        """
        a0 = 1e-6
        envelope = np.exp(-((t - 8) ** 8.0) / 4.0 ** 8.0)

        return 0 * envelope * a0 * np.cos(k0 * x + w0 * t)

    field_store = np.zeros([nt, nx])
    dist_store = np.zeros([nt, nx, nv])
    t_store = np.zeros(nt)

    e = field.get_total_electric_field(driver_function(x, t[0]), f=f, dv=dv, kx=kx)

    it = 0
    t_store[it] = t[it]
    dist_store[it] = f
    field_store[it] = e

    for it in range(1, nt):
        e, f = step.full_PEFRL_ps_step(
            f, x, kx, v, kv, dv, t[it], dt, e, driver_function
        )

        t_store[it] = t[it]
        dist_store[it] = f
        field_store[it] = e

    np.testing.assert_almost_equal(field_store[0], field_store[-1], decimal=2)
    np.testing.assert_almost_equal(dist_store[0], dist_store[-1], decimal=2)
