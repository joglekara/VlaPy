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


def get_full_leapfrog_step(vdfdx, edfdv, field_solve, dt, driver_function):
    """
    This function returns the leapfrog step pieced together with the provided components

    :param vdfdx: (function) the chosen v df/dx integrator
    :param edfdv: (function) the chosen e df/dv integrator
    :param field_solve: (function) the chosen field solver
    :param dt: (float) the time-step
    :param driver_function: (function) the function providing the driver
    :return: (function) above inputs initialized as static variables
    """

    def full_leapfrog_ps_step(e, f, t):
        """
        Takes a step forward in time for f and e

        Uses leapfrog scheme
        1 - spatial advection for 0.5 dt

        2a - field solve
        2b - velocity advection for dt

        3 - spatial advection for 0.5 dt

        :param e: (float array (nx, )) electric field
        :param f: (float array (nx, nv)) distribution function
        :param t: (float) current time
        :return: (float array (nx, nv)) updated distribution function
        """
        f = edfdv(f=f, e=e, dt=0.5 * dt)
        f = vdfdx(f=f, dt=dt)
        e = field_solve(driver_field=driver_function(t + dt), f=f)
        f = edfdv(f=f, e=e, dt=0.5 * dt)

        return e, f

    return full_leapfrog_ps_step


def get_full_pefrl_step(vdfdx, edfdv, field_solve, dt, driver_function):
    """
    This function returns the Performance-Extended Forest-Ruth-Like
    step [1-2] pieced together with the provided components

    [1] - Omelyan, I. P., Mryglod, I. M., & Folk, R. (2002). Optimized Forest–Ruth- and Suzuki-like
    algorithms for integration of motion in many-body systems.
    Computer Physics Communications, 146(2), 188–202. https://doi.org/10.1016/S0010-4655(02)00451-4

    [2] - http://physics.ucsc.edu/~peter/242/leapfrog.pdf

    :param vdfdx: (function) the chosen v df/dx integrator
    :param edfdv: (function) the chosen e df/dv integrator
    :param field_solve: (function) the chosen field solver
    :param dt: (float) the time-step
    :param driver_function: (function) the function providing the driver
    :return: (function) above inputs initialized as static variables
    """

    def full_pefrl_ps_step(e, f, t):
        """
        Takes a step forward in time for f and e using the
        Performance-Extended Forest-Ruth-Like algorithm [1-2]

        This is a 4th order symplectic integrator.

        [1] - Omelyan, I. P., Mryglod, I. M., & Folk, R. (2002). Optimized Forest–Ruth- and Suzuki-like
        algorithms for integration of motion in many-body systems.
        Computer Physics Communications, 146(2), 188–202. https://doi.org/10.1016/S0010-4655(02)00451-4

        [2] - http://physics.ucsc.edu/~peter/242/leapfrog.pdf

        :param e: (float array (nx, )) electric field
        :param f: (float array (nx, nv)) distribution function
        :param t: (float) current time
        :return: (float array (nx, nv)) updated distribution function
        """
        xsi = 0.1786178958448091
        lambd = -0.2123418310626054
        chi = -0.6626458266981849e-1

        dt1 = xsi * dt
        dt2 = chi * dt
        dt3 = (1.0 - 2.0 * (chi + xsi)) * dt
        dt4 = dt2
        dt5 = dt1

        vdt1 = 0.5 * (1.0 - 2.0 * lambd) * dt
        vdt2 = lambd * dt
        vdt3 = vdt2
        vdt4 = vdt1

        f = vdfdx(f, dt1)
        e = field_solve(driver_function(t + dt1), f=f)

        # v1
        f = edfdv(f, e, vdt1)

        # x2
        f = vdfdx(f, dt2)
        e = field_solve(driver_function(t + dt1 + dt2), f=f)

        # v2
        f = edfdv(f, e, vdt2)

        # x3
        f = vdfdx(f, dt3)
        e = field_solve(driver_function(t + dt1 + dt2 + dt3), f=f)

        # v3
        f = edfdv(f, e, vdt3)

        # x4
        f = vdfdx(f, dt4)
        e = field_solve(
            driver_function(t + dt1 + dt2 + dt3 + dt4),
            f=f,
        )

        # v4
        f = edfdv(f, e, vdt4)

        # x5
        f = vdfdx(f, dt5)
        e = field_solve(
            driver_function(t + dt1 + dt2 + dt3 + dt4 + dt5),
            f=f,
        )

        return e, f

    return full_pefrl_ps_step


def get_6th_order_integrator(vdfdx, edfdv, field_solve, dt, driver_function):
    """
    This function returns the 6th order step [1] pieced together with the provided components


    [1] - Casas, F., Crouseilles, N., Faou, E., & Mehrenberger, M. (2017). High-order Hamiltonian splitting
    for the Vlasov–Poisson equations.
    Numerische Mathematik, 135(3), 769–801. https://doi.org/10.1007/s00211-016-0816-z

    :param vdfdx: (function) the chosen v df/dx integrator
    :param edfdv: (function) the chosen e df/dv integrator
    :param field_solve: (function) the chosen field solver
    :param dt: (float) the time-step
    :param driver_function: (function) the function providing the driver
    :return: (function) above inputs initialized as static variables
    """

    a1 = 0.168735950563437422448196
    a2 = 0.377851589220928303880766
    a3 = -0.093175079568731452657924
    b1 = 0.049086460976116245491441
    b2 = 0.264177609888976700200146
    b3 = 0.186735929134907054308413
    c1 = -0.000069728715055305084099
    c2 = -0.000625704827430047189169
    c3 = -0.002213085124045325561636
    d2 = -2.916600457689847816445691e-6
    d3 = 3.048480261700038788680723e-5
    e3 = 4.985549387875068121593988e-7

    def sixth_order_step(e, f, t):
        """
        This is the 6th order integrator for 1D Vlasov-Poisson systems given in
        Casas, F., Crouseilles, N., Faou, E., & Mehrenberger, M. (2017). High-order Hamiltonian splitting for
        the Vlasov–Poisson equations.
        Numerische Mathematik, 135(3), 769–801. https://doi.org/10.1007/s00211-016-0816-z

        :param e: (float array (nx, )) electric field
        :param f: (float array (nx, nv)) distribution function
        :param t: (float) current time
        :return: (float array (nx, nv)) updated distribution function
        """
        D1 = b1 + 2.0 * c1 * dt ** 2.0
        D2 = b2 + 2.0 * c2 * dt ** 2.0 + 4.0 * d2 * dt ** 4.0
        D3 = b3 + 2.0 * c3 * dt ** 2.0 + 4.0 * d3 * dt ** 4.0 - 8.0 * e3 * dt ** 6.0

        f = edfdv(f=f, e=e, dt=D1 * dt)

        f = vdfdx(f=f, dt=a1 * dt)
        e = field_solve(driver_field=driver_function(t + a1 * dt), f=f)

        f = edfdv(f=f, e=e, dt=D2 * dt)

        f = vdfdx(f=f, dt=a2 * dt)
        e = field_solve(driver_field=driver_function(t + a2 * dt), f=f)

        f = edfdv(f=f, e=e, dt=D3 * dt)

        f = vdfdx(f=f, dt=a3 * dt)
        e = field_solve(driver_field=driver_function(t + a3 * dt), f=f)

        f = edfdv(f=f, e=e, dt=D3 * dt)

        f = vdfdx(f=f, dt=a2 * dt)
        e = field_solve(driver_field=driver_function(t + a2 * dt), f=f)

        f = edfdv(f=f, e=e, dt=D2 * dt)

        f = vdfdx(f=f, dt=a1 * dt)
        e = field_solve(driver_field=driver_function(t + a1 * dt), f=f)

        f = edfdv(f=f, e=e, dt=D1 * dt)

        return e, f

    return sixth_order_step


def get_time_integrator(
    time_integrator_name, vdfdx, edfdv, field_solver, stuff_for_time_loop
):
    """
    This function returns the full time-integrator pieced together with the provided
    components of the integrator

    :param time_integrator_name: (string) name of the time-integrator chosen in the input parameters
    :param vdfdx: (function) the chosen v df/dx integrator
    :param edfdv: (function) the chosen e df/dv integrator
    :param field_solve: (function) the chosen field solver
    :param stuff_for_time_loop: (dictionary) derived parameters for simulation
    :return: (function) function that updates the distribution function with the
    above inputs initialized as static variables
    """

    if time_integrator_name == "leapfrog":
        vp_step = get_full_leapfrog_step(
            vdfdx=vdfdx,
            edfdv=edfdv,
            field_solve=field_solver,
            dt=stuff_for_time_loop["dt"],
            driver_function=stuff_for_time_loop["driver_function"],
        )
    elif time_integrator_name == "pefrl":
        vp_step = get_full_pefrl_step(
            vdfdx=vdfdx,
            edfdv=edfdv,
            field_solve=field_solver,
            dt=stuff_for_time_loop["dt"],
            driver_function=stuff_for_time_loop["driver_function"],
        )
    elif time_integrator_name == "h-sixth":
        vp_step = get_6th_order_integrator(
            vdfdx=vdfdx,
            edfdv=edfdv,
            field_solve=field_solver,
            dt=stuff_for_time_loop["dt"],
            driver_function=stuff_for_time_loop["driver_function"],
        )
    else:
        raise NotImplementedError(
            "df/dt : <" + time_integrator_name + "> has not yet been implemented"
        )

    return vp_step
