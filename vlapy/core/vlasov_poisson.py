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

        f = edfdv(f=f, e=e, dt=0.5 * dt)
        f = vdfdx(f=f, dt=dt)
        e = field_solve(driver_field=driver_function(t + dt), f=f)
        f = edfdv(f=f, e=e, dt=0.5 * dt)

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


def get_time_integrator(
    time_integrator_name, vdfdx, edfdv, field_solver, stuff_for_time_loop
):
    if time_integrator_name == "leapfrog":
        vp_step = get_full_leapfrog_step(
            vdfdx=vdfdx,
            edfdv=edfdv,
            field_solve=field_solver,
            x=stuff_for_time_loop["x"],
            dt=stuff_for_time_loop["dt"],
            driver_function=stuff_for_time_loop["driver_function"],
        )
    elif time_integrator_name == "pefrl":
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
