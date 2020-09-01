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
from scipy import interpolate, fft


def _get_padded_grid_(ax):
    """
    This function returns a padded axis with periodic boundaries in place

    :param ax: (float array) the axis to pad
    :return: (float array) the padded axis
    """
    ax_pad = np.zeros(ax.size + 2)
    ax_pad[1:-1] = ax
    ax_pad[0] = ax[0] - (ax[2] - ax[1])
    ax_pad[-1] = ax[-1] + (ax[2] - ax[1])

    return ax_pad


def get_vdfdx_sl(x, v):
    """
    Get the v df/dx Semi-Lagrangian stepper

    :param x: (float array (nx, )) the spatial grid
    :param v: (float array (nv, )) the velocity grid
    :return: a function with the above inputs used to initialize useful static variables
    """

    xm, vm = np.meshgrid(x, v, indexing="ij")
    xm = xm.flatten()
    vm = vm.flatten()

    x_pad = _get_padded_grid_(x)
    f_pad = np.zeros((x.size + 2, v.size))

    def update_spatial_adv_sl(f, dt):
        """
        evolution of df/dt = v df/dx using the Backward Semi-Lagrangian method popularized by
        [1] and widely used since.

        [1] - Cheng, C. ., & Knorr, G. (1976). The integration of the vlasov equation in configuration space.
        Journal of Computational Physics, 22(3), 330–351. https://doi.org/10.1016/0021-9991(76)90053-X

        :param f: (float array (nx, nv)) distribution function
        :param dt: (float) timestep
        :return: (float array (nx, nv)) updated distribution function
        """

        f_pad[1:-1, :] = f
        f_pad[0, :] = f[-1, :]
        f_pad[-1, :] = f[0, :]

        f_interpolator = interpolate.RectBivariateSpline(x_pad, v, f_pad)
        f_out = f_interpolator(xm - vm * dt, vm, grid=False).reshape((x.size, v.size))

        return f_out

    return update_spatial_adv_sl


def get_vdfdx_exponential(kx, v):
    """
    This function creates the exponential v df/dx stepper

    It uses kx and v as metadata that should stay constant throughout the simulation

    :param kx: (float array (nx, )) the real-space wavenumber
    :param v: (float array  (nv, )) the velocity grid
    :return: a function with the above values initialized as static variables
    """

    def step_vdfdx_exponential(f, dt):
        """
        evolution of df/dt = v df/dx using the exponential integrator described in
        [1]

        [1] - https://juliavlasov.github.io/ -- Dr. Pierre Navarro

        :param f: (float array (nx, nv)) distribution function
        :param dt: (float) timestep
        :return: (float array (nx, nv)) updated distribution function
        """

        return np.real(
            fft.ifft(np.exp(-1j * kx[:, None] * dt * v) * fft.fft(f, axis=0), axis=0)
        )

    return step_vdfdx_exponential


def get_edfdv_exponential(kv):
    """
    This function creates the exponential v df/dx stepper

    It uses kv as metadata that should stay constant throughout the simulation

    :param kv: (float array (nv, )) the velocity-space wavenumber
    :return: a function with the above values initialized as static variables
    """

    def step_edfdv_exponential(f, e, dt):
        """
        evolution of df/dt = e df/dv using the exponential integrator described in
        [1].

        [1] - https://juliavlasov.github.io/ -- Dr. Pierre Navarro

        :param f: (float array (nx, nv)) distribution function
        :param e: (float array (nx, )) the electric field in real space
        :param dt: (float) timestep
        :return: (float array (nx, nv)) updated distribution function
        """

        return np.real(
            fft.ifft(np.exp(-1j * kv * dt * e[:, None]) * fft.fft(f, axis=1), axis=1)
        )

    return step_edfdv_exponential


def get_edfdv_center_differenced(dv):
    """
    This function creates the center differenced e df/dv stepper

    It uses dv as metadata that should stay constant throughout the simulation

    :param dv: (float) the velocity grid spacing
    :return: a function with the above values initialized as static variables
    """

    def step_edfdv_center_difference(f, e, dt):
        """
        This method calculates the f + dt * e * df/dv using naive
        2nd-order center differencing

        :param f: (float array (nx, nv)) distribution function
        :param e: (float array (nx, )) the electric field in real space
        :param dt: (float) timestep
        :return: (float array (nx, nv)) updated distribution function
        """
        return f - e[:, None] * np.gradient(f, dv, axis=1, edge_order=2) * dt

    return step_edfdv_center_difference


def get_edfdv_sl(x, v):
    """
    Get the e df/dv Semi-Lagrangian stepper

    :param x: (float array (nx, )) the spatial grid
    :param v: (float array (nv, )) the velocity grid
    :return: a function with the above inputs used to initialize useful static variables
    """

    xm, vm = np.meshgrid(x, v, indexing="ij")
    xm = xm.flatten()
    vm = vm.flatten()

    v_pad = _get_padded_grid_(v)
    f_pad = np.zeros((x.size, v.size + 2))

    def update_velocity_adv_sl(f, e, dt):
        """
        evolution of df/dt = e df/dv according to the Backward Semi-Lagrangian technique popularized by [1]

        [1] - Cheng, C. ., & Knorr, G. (1976). The integration of the vlasov equation in configuration space.
        Journal of Computational Physics, 22(3), 330–351. https://doi.org/10.1016/0021-9991(76)90053-X

        :param f: distribution function. (numpy array of shape (nx, nv))
        :param e: electric field (numpy array of shape (nx,))
        :param dt: timestep (single float value)
        :return:
        """

        f_pad[:, 1:-1] = f
        f_pad[:, 0] = f[:, -1]
        f_pad[:, -1] = f[:, 0]

        e_fit = interpolate.interp1d(x, e, kind="cubic")

        em = e_fit(xm)

        f_interpolator = interpolate.RectBivariateSpline(x, v_pad, f_pad)
        f_out = f_interpolator(xm, vm - em * dt, grid=False).reshape((x.size, v.size))

        return f_out

    return update_velocity_adv_sl


def get_vdfdx(stuff_for_time_loop, vdfdx_implementation="exponential"):
    """
    This function enables VlaPy to choose the implementation of the vdfdx stepper
    to use in the lower level sections of the simulation

    :param stuff_for_time_loop: (dictionary) contains the derived parameters for the simulation
    :param vdfdx_implementation: (string) the chosen v df/dx implementation for for this simulation
    :return:
    """
    if vdfdx_implementation == "exponential":
        vdfdx = get_vdfdx_exponential(
            kx=stuff_for_time_loop["kx"], v=stuff_for_time_loop["v"]
        )
    elif vdfdx_implementation == "sl":
        vdfdx = get_vdfdx_sl(x=stuff_for_time_loop["x"], v=stuff_for_time_loop["v"])
    else:
        raise NotImplementedError(
            "v df/dx: <"
            + vdfdx_implementation
            + "> has not yet been implemented in NumPy/SciPy"
        )

    return vdfdx


def get_edfdv(stuff_for_time_loop, edfdv_implementation="exponential"):
    """
    This function enables VlaPy to choose the implementation of the edfdv stepper
    to use in the lower level sections of the simulation

    :param stuff_for_time_loop: (dictionary) contains the derived parameters for the simulation
    :param vdfdx_implementation: (string) the chosen v df/dx implementation for for this simulation
    :return:
    """
    if edfdv_implementation == "exponential":
        edfdv = get_edfdv_exponential(kv=stuff_for_time_loop["kv"])
    elif edfdv_implementation == "cd2":
        edfdv = get_edfdv_center_differenced(dv=stuff_for_time_loop["dv"])
    elif edfdv_implementation == "sl":
        edfdv = get_edfdv_sl(v=stuff_for_time_loop["v"], x=stuff_for_time_loop["x"])
    else:
        raise NotImplementedError(
            "e df/dv: <"
            + edfdv_implementation
            + "> has not yet been implemented in NumPy/SciPy"
        )

    return edfdv
