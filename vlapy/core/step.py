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

from vlapy.core import field, vlasov, collisions


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


def get_full_leapfrog_step(vdfdx, edfdv, field_solve, x, dt, driver_function):
    def full_leapfrog_ps_step(e, f, t):
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

        f = edfdv(f, e, 0.5 * dt)
        f = vdfdx(f, dt)
        e = field_solve(driver_function(t + dt), f=f)
        f = edfdv(f, e, 0.5 * dt)

        return e, f

    return full_leapfrog_ps_step


def get_full_pefrl_step(vdfdx, edfdv, field_solve, x, kx, v, kv, dt, driver_function):
    def full_pefrl_ps_step(e, f, t):
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
        # f = vlasov.update_spatial_adv_spectral(f, kx, v, dt1)
        # e = field.get_total_electric_field(
        #     driver_function(x, t + dt1), f=f, dv=dv, one_over_kx=one_over_kx
        # )
        #
        # # v1
        # f = vlasov.update_velocity_adv_spectral(
        #     f, kv, e, 0.5 * (1.0 - 2.0 * lambd) * dt
        # )
        #
        # # x2
        # f = vlasov.update_spatial_adv_spectral(f, kx, v, dt2)
        # e = field.get_total_electric_field(
        #     driver_function(x, t + dt1 + dt2), f=f, dv=dv, one_over_kx=one_over_kx
        # )
        #
        # # v2
        # f = vlasov.update_velocity_adv_spectral(f, kv, e, lambd * dt)
        #
        # # x3
        # f = vlasov.update_spatial_adv_spectral(f, kx, v, dt3)
        # e = field.get_total_electric_field(
        #     driver_function(x, t + dt1 + dt2 + dt3), f=f, dv=dv, one_over_kx=one_over_kx
        # )
        #
        # # v3
        # f = vlasov.update_velocity_adv_spectral(f, kv, e, lambd * dt)
        #
        # # x4
        # f = vlasov.update_spatial_adv_spectral(f, kx, v, dt4)
        # e = field.get_total_electric_field(
        #     driver_function(x, t + dt1 + dt2 + dt3 + dt4),
        #     f=f,
        #     dv=dv,
        #     one_over_kx=one_over_kx,
        # )
        #
        # # v4
        # f = vlasov.update_velocity_adv_spectral(
        #     f, kv, e, 0.5 * (1.0 - 2.0 * lambd) * dt
        # )
        #
        # # x5
        # f = vlasov.update_spatial_adv_spectral(f, kx, v, dt5)
        # e = field.get_total_electric_field(
        #     driver_function(x, t + dt1 + dt2 + dt3 + dt4 + dt5),
        #     f=f,
        #     dv=dv,
        #     one_over_kx=one_over_kx,
        # )

        f = vdfdx(f, dt1)
        e = field_solve(driver_function(t + dt1), f=f)

        # v1
        f = edfdv(f, kv, e, 0.5 * (1.0 - 2.0 * lambd) * dt)

        # x2
        f = vdfdx(f, kx, v, dt2)
        e = field_solve(driver_function(t + dt1 + dt2), f=f)

        # v2
        f = edfdv(f, kv, e, lambd * dt)

        # x3
        f = vdfdx(f, dt3)
        e = field_solve(driver_function(t + dt1 + dt2 + dt3), f=f)

        # v3
        f = edfdv(f, kv, e, lambd * dt)

        # x4
        f = vdfdx(f, dt4)
        e = field_solve(driver_function(t + dt1 + dt2 + dt3 + dt4), f=f,)

        # v4
        f = edfdv(f, e, 0.5 * (1.0 - 2.0 * lambd) * dt)

        # x5
        f = vdfdx(f, dt5)
        e = field_solve(driver_function(t + dt1 + dt2 + dt3 + dt4 + dt5), f=f,)

        return e, f

    return full_pefrl_ps_step


def get_vlasov_poisson_step(all_params, stuff_for_time_loop):
    """
    This is the highest level function for getting a jitted Vlasov-Poisson.

    :param static_args:
    :return:
    """

    if all_params["vlasov-poisson"]["vdfdx"] == "exponential":
        vdfdx = vlasov.get_vdfdx_exponential(
            kx=stuff_for_time_loop["kx"], v=stuff_for_time_loop["v"]
        )
    else:
        raise NotImplementedError

    if all_params["vlasov-poisson"]["edfdv"] == "exponential":
        edfdv = vlasov.get_edfdv_exponential(kv=stuff_for_time_loop["kv"])
    else:
        raise NotImplementedError

    if all_params["vlasov-poisson"]["poisson"] == "spectral":
        field_solver = field.get_field_solver(
            dv=stuff_for_time_loop["dv"], one_over_kx=stuff_for_time_loop["one_over_kx"]
        )
    else:
        raise NotImplementedError

    if all_params["vlasov-poisson"]["time"] == "leapfrog":
        vp_step = get_full_leapfrog_step(
            vdfdx=vdfdx,
            edfdv=edfdv,
            field_solve=field_solver,
            x=stuff_for_time_loop["x"],
            dt=stuff_for_time_loop["dt"],
            driver_function=stuff_for_time_loop["driver_function"],
        )
    elif all_params["vlasov-poisson"]["time"] == "pefrl":
        vp_step = get_full_pefrl_step(
            vdfdx=vdfdx,
            edfdv=edfdv,
            field_solve=field_solver,
            x=stuff_for_time_loop["x"],
            kx=stuff_for_time_loop["kx"],
            v=stuff_for_time_loop["v"],
            kv=stuff_for_time_loop["kv"],
            dt=stuff_for_time_loop["dt"],
            driver_function=stuff_for_time_loop["driver_function"],
        )
    else:
        raise NotImplementedError

    return vp_step


def get_collision_step(stuff_for_time_loop, all_params):

    solver = collisions.get_matrix_solver(
        nx=stuff_for_time_loop["nx"],
        nv=stuff_for_time_loop["nv"],
        solver_name=all_params["fokker-planck"]["solver"],
    )
    get_collision_matrix_for_all_x = collisions.get_batched_array_maker(
        vax=stuff_for_time_loop["v"],
        nv=stuff_for_time_loop["nv"],
        nx=stuff_for_time_loop["nx"],
        nu=stuff_for_time_loop["nu"],
        dt=stuff_for_time_loop["dt"],
        dv=stuff_for_time_loop["dv"],
        operator=all_params["fokker-planck"]["type"],
    )

    def take_collision_step(f):

        # The three diagonals representing collision operator for all x
        cee_a, cee_b, cee_c = get_collision_matrix_for_all_x(f_xv=f)

        # Solve over all x
        return solver(cee_a, cee_b, cee_c, f)

    return take_collision_step


def get_storage_step(stuff_for_time_loop):
    dv = stuff_for_time_loop["dv"]
    v = stuff_for_time_loop["v"]

    def storage_step(temp_health_storage, e, de, f, i):

        # Density
        temp_health_storage["mean_n"][i] = np.mean(np.trapz(f, dx=dv, axis=1), axis=0)

        # Momentum
        temp_health_storage["mean_v"][i] = np.mean(
            np.trapz(f ** v, dx=dv, axis=1), axis=0
        )

        # Energy
        temp_health_storage["mean_T"][i] = np.mean(
            np.trapz(f ** v ** 2.0, dx=dv, axis=1), axis=0
        )
        temp_health_storage["mean_e2"][i] = np.mean(e ** 2.0, axis=0)
        temp_health_storage["mean_de2"][i] = np.mean(de ** 2.0, axis=0)
        temp_health_storage["mean_t_plus_e2_minus_de2"][i] = temp_health_storage[
            "mean_T"
        ][i] + (temp_health_storage["mean_e2"][i] - temp_health_storage["mean_de2"][i])
        temp_health_storage["mean_t_plus_e2_plus_de2"][i] = (
            temp_health_storage["mean_T"][i]
            + temp_health_storage["mean_e2"][i]
            + temp_health_storage["mean_de2"][i]
        )

        # Abstract
        temp_health_storage["mean_f2"][i] = np.mean(
            np.trapz(f ** 2.0, dx=dv, axis=1), axis=0
        )
        temp_health_storage["mean_flogf"][i] = np.mean(
            np.trapz(f * np.log(f), dx=dv, axis=1), axis=0
        )

        return temp_health_storage

    return storage_step


def get_timestep(all_params, stuff_for_time_loop):
    """
    Gets the full VFP + logging timestep

    :param all_params:
    :return:
    """

    vp_step = get_vlasov_poisson_step(
        all_params=all_params, stuff_for_time_loop=stuff_for_time_loop
    )
    fp_step = get_collision_step(
        all_params=all_params, stuff_for_time_loop=stuff_for_time_loop
    )
    storage_step = get_storage_step(stuff_for_time_loop=stuff_for_time_loop)

    def timestep(temp_storage, i):
        """
        This is just a wrapper around a Vlasov-Poisson + Fokker-Planck timestep

        :param val:
        :param i:
        :return:
        """
        e = temp_storage["e"]
        f = temp_storage["f"]
        t = temp_storage["time_batch"][i]
        de = temp_storage["driver_array_batch"][i]

        e, f = vp_step(e=e, f=f, t=t)
        f = fp_step(f=f)

        temp_storage["health"] = storage_step(temp_storage["health"], e, de, f, i)

        temp_storage["e"] = e
        temp_storage["f"] = f

        return temp_storage, i

    return timestep
