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
from scipy import fft

from vlapy.core import field, vlasov, collisions, vlasov_poisson


def get_vlasov_poisson_step(all_params, stuff_for_time_loop):
    """
    This is the highest level function for getting a jitted Vlasov-Poisson.

    :param static_args:
    :return:
    """

    vdfdx = vlasov.get_vdfdx(
        stuff_for_time_loop=stuff_for_time_loop,
        vdfdx_implementation=all_params["vlasov-poisson"]["vdfdx"],
    )
    edfdv = vlasov.get_edfdv(
        stuff_for_time_loop=stuff_for_time_loop,
        edfdv_implementation=all_params["vlasov-poisson"]["edfdv"],
    )

    field_solver = field.get_field_solver(
        stuff_for_time_loop=stuff_for_time_loop,
        field_solver_implementation=all_params["vlasov-poisson"]["poisson"],
    )

    vp_step = vlasov_poisson.get_time_integrator(
        time_integrator_name=all_params["vlasov-poisson"]["time"],
        vdfdx=vdfdx,
        edfdv=edfdv,
        field_solver=field_solver,
        stuff_for_time_loop=stuff_for_time_loop,
    )

    return vp_step


def get_collision_step(stuff_for_time_loop, all_params):

    if all_params["nu"] == 0.0:

        def take_collision_step(f):
            return f

    elif all_params["nu"] > 0.0:

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

    else:
        raise NotImplementedError

    return take_collision_step


def get_f_update(store_f_rule):
    """
    This function returns the function used in the stepper for storing f in a batch
    It performs the necessary transformations if we're just storing Fourier modes.

    :param store_f_rule:
    :return:
    """
    if store_f_rule["space"] == "all":

        def get_f_to_store(f):
            return f

    elif store_f_rule["space"][0] == "k0":

        def get_f_to_store(f):
            return fft.fft(f, axis=0)[
                : len(store_f_rule),
            ]

    else:
        raise NotImplementedError

    return get_f_to_store


def get_fields_update(dv, v):
    def update_fields(temp_storage_fields, e, de, f, i):
        temp_storage_fields["e"][i] = e
        temp_storage_fields["driver"][i] = de
        temp_storage_fields["n"][i] = np.trapz(f, dx=dv, axis=1)
        temp_storage_fields["j"][i] = np.trapz(f * v, dx=dv, axis=1)
        temp_storage_fields["T"][i] = np.trapz(f * v ** 2, dx=dv, axis=1)
        temp_storage_fields["q"][i] = np.trapz(f * v ** 3, dx=dv, axis=1)
        temp_storage_fields["fv4"][i] = np.trapz(f * v ** 4, dx=dv, axis=1)
        temp_storage_fields["vN"][i] = np.trapz(f * v ** 5, dx=dv, axis=1)

        return temp_storage_fields

    return update_fields


def get_series_update(dv):
    def update_series(temp_storage, e, de, f, i):
        temp_storage_series = temp_storage["series"]
        # Density
        temp_storage_series["mean_n"][i] = np.mean(
            temp_storage["fields"]["n"][i], axis=0
        )

        # Momentum
        temp_storage_series["mean_j"][i] = np.mean(
            temp_storage["fields"]["j"][i], axis=0
        )

        # Energy
        temp_storage_series["mean_T"][i] = np.mean(
            temp_storage["fields"]["T"][i], axis=0
        )
        temp_storage_series["mean_e2"][i] = np.mean(e ** 2.0, axis=0)
        temp_storage_series["mean_de2"][i] = np.mean(de ** 2.0, axis=0)

        # Abstract
        temp_storage_series["mean_f2"][i] = np.mean(
            np.trapz(np.real(f) ** 2.0, dx=dv, axis=1), axis=0
        )
        temp_storage_series["mean_flogf"][i] = np.mean(
            np.trapz(f * np.log(f), dx=dv, axis=1), axis=0
        )

        return temp_storage_series

    return update_series


def get_storage_step(stuff_for_time_loop):
    dv = stuff_for_time_loop["dv"]
    v = stuff_for_time_loop["v"]

    store_f_function = get_f_update(
        store_f_rule=stuff_for_time_loop["rules_to_store_f"]
    )

    update_fields = get_fields_update(dv=dv, v=v)
    update_series = get_series_update(dv=dv)

    def storage_step(temp_storage, e, de, f, i):
        temp_storage["stored_f"][i] = store_f_function(f)

        temp_storage["e"] = e
        temp_storage["f"] = f

        temp_storage["fields"] = update_fields(
            temp_storage_fields=temp_storage["fields"], de=de, e=e, f=f, i=i
        )
        temp_storage["series"] = update_series(
            temp_storage=temp_storage, f=f, de=de, e=e, i=i
        )

        return temp_storage

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

        temp_storage = storage_step(temp_storage=temp_storage, e=e, de=de, f=f, i=i)

        return temp_storage, i

    return timestep
