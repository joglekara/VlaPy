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


from time import time
from tqdm import tqdm

from vlapy import initializers, field_driver
from vlapy.core import step


def get_sim_config_and_inner_loop_step(
    all_params,
    stuff_for_time_loop,
    steps_in_loop,
    rules_to_store_f,
):
    """
    This is the function that gets the correct inner loop calculation routines given
    the backend

    :param all_params:
    :param stuff_for_time_loop:
    :param steps_in_loop:
    :param rules_to_store_f:
    :param type:
    :return:
    """

    if all_params["backend"]["core"] == "numpy":
        import numpy as np_for_time_loop
    else:
        raise NotImplementedError

    do_inner_loop = get_inner_loop_stepper(
        all_params, stuff_for_time_loop, steps_in_loop
    )

    sim_config = get_arrays_for_inner_loop(
        stuff_for_time_loop, steps_in_loop, rules_to_store_f, this_np=np_for_time_loop
    )

    return sim_config, do_inner_loop


def get_everything_ready_for_outer_loop(
    diagnostics, all_params, pulse_dictionary, overall_num_steps
):
    """
    In order to keep the main time loop clean, this function handles all the
    necessary initialization and array creation for the main simulation time loop.

    Initialized here:
    spatial grid
    velocity grid
    distribution function
    time grid
    driver array

    :param diagnostics:
    :param all_params:
    :param pulse_dictionary:
    :return:
    """

    import numpy as np

    # Log desired parameters
    initializers.log_initial_conditions(
        all_params=all_params,
        pulse_dictionary=pulse_dictionary,
    )

    # Initialize machinery
    # Distribution function
    f = initializers.initialize_distribution(
        nx=all_params["nx"], nv=all_params["nv"], vmax=all_params["vmax"]
    )

    # Spatial Grid
    dx, x, kx, one_over_kx = initializers.initialize_spatial_quantities(
        xmin=all_params["xmin"], xmax=all_params["xmax"], nx=all_params["nx"]
    )

    # Velocity grid
    dv, v, kv = initializers.initialize_velocity_quantities(
        vmax=all_params["vmax"], nv=all_params["nv"]
    )

    t_dummy = np.linspace(0, all_params["tmax"], all_params["nt"])
    dt = t_dummy[1] - t_dummy[0]
    t = dt * np.arange(overall_num_steps)

    # Field Driver
    driver_function = field_driver.get_driver_function(
        x=x, pulse_dictionary=pulse_dictionary, np=np
    )

    driver_array = field_driver.make_driver_array(
        function=driver_function, x_axis=x, time_axis=t
    )

    everything_for_time_loop = {
        "e": np.zeros(x.size),
        "f": f,
        "nx": all_params["nx"],
        "kx": kx,
        "x": x,
        "one_over_kx": one_over_kx,
        "v": v,
        "kv": kv,
        "nv": all_params["nv"],
        "dv": dv,
        "driver": driver_array,
        "driver_function": driver_function,
        "dt": dt,
        "nu": all_params["nu"],
        "t": t,
        "rules_to_store_f": diagnostics.rules_to_store_f,
        "vlasov-poisson": all_params["vlasov-poisson"],
        "fokker-planck": all_params["fokker-planck"],
    }

    return everything_for_time_loop


def get_arrays_for_inner_loop(stuff_for_time_loop, nt_in_loop, store_f_rules, this_np):
    """
    This function converts the previously created NumPy arrays to NumPy arrays.

    It also creates the temporary storage for e and f that is used for the
    low-level time-loop

    :param stuff_for_time_loop:
    :param nt_in_loop:
    :param store_f_rules:
    :return:
    """
    import numpy as np

    # This is where we initialize the right distribution function storage array
    # If we're saving all the "x" values then that array is created
    if store_f_rules["space"] == "all":
        store_f = np.zeros(
            (nt_in_loop,) + stuff_for_time_loop["f"].shape, dtype=np.float64
        )
        store_f[
            0,
        ] = stuff_for_time_loop["f"]

    # If we're saving the first M spatial-modes then that array is created
    elif isinstance(store_f_rules["space"], list) and store_f_rules["space"][0] == "k0":
        store_f = np.zeros(
            (
                nt_in_loop,
                len(store_f_rules["space"]),
                stuff_for_time_loop["f"].shape[1],
            ),
            dtype=np.complex64,
        )

        store_f[0,] = np.fft.fft(
            stuff_for_time_loop["f"], axis=0
        )[: len(store_f_rules["space"])]

    else:
        raise NotImplementedError

    # This is where we return the whole simulation configuration dictionary
    return {
        "time_batch": this_np.zeros(nt_in_loop),
        "e": this_np.array(stuff_for_time_loop["e"]),
        "f": this_np.array(stuff_for_time_loop["f"]),
        "stored_f": this_np.array(store_f),
        "mean_cum_de2_previous": 0.0,
        "series": {
            "mean_n": this_np.zeros(nt_in_loop),
            "mean_j": this_np.zeros(nt_in_loop),
            "mean_T": this_np.zeros(nt_in_loop),
            "mean_e2": this_np.zeros(nt_in_loop),
            "mean_de2": this_np.zeros(nt_in_loop),
            "mean_f2": this_np.zeros(nt_in_loop),
            "mean_flogf": this_np.zeros(nt_in_loop),
        },
        "fields": {
            "e": this_np.zeros((nt_in_loop,) + stuff_for_time_loop["e"].shape),
            "driver": this_np.zeros((nt_in_loop,) + stuff_for_time_loop["e"].shape),
            "n": this_np.zeros((nt_in_loop,) + stuff_for_time_loop["e"].shape),
            "j": this_np.zeros((nt_in_loop,) + stuff_for_time_loop["e"].shape),
            "T": this_np.zeros((nt_in_loop,) + stuff_for_time_loop["e"].shape),
            "q": this_np.zeros((nt_in_loop,) + stuff_for_time_loop["e"].shape),
            "fv4": this_np.zeros((nt_in_loop,) + stuff_for_time_loop["e"].shape),
            "vN": this_np.zeros((nt_in_loop,) + stuff_for_time_loop["e"].shape),
        },
    }


def post_inner_loop_update(temp_storage, t0, this_np):
    temp_storage["time_for_batch"] = time() - t0

    # Energy from driver
    temp_storage["series"]["mean_cum_de2"] = temp_storage[
        "mean_cum_de2_previous"
    ] + this_np.cumsum(temp_storage["series"]["mean_de2"])

    # See if E_driver = E_f + E_e
    temp_storage["series"]["mean_t_plus_e2_minus_cum_de2"] = temp_storage["series"][
        "mean_T"
    ] + (temp_storage["series"]["mean_e2"] - temp_storage["series"]["mean_cum_de2"])

    temp_storage["series"]["mean_t_plus_e2_plus_cum_de2"] = (
        temp_storage["series"]["mean_T"]
        + temp_storage["series"]["mean_e2"]
        + temp_storage["series"]["mean_de2"]
    )

    temp_storage["mean_cum_de2_previous"] = temp_storage["series"]["mean_cum_de2"][-1]

    return temp_storage


def get_inner_loop_stepper(all_params, stuff_for_time_loop, steps_in_loop):
    """
    This is where the `NumPy`-based inner-loop is created and returned

    :param all_params:
    :param stuff_for_time_loop:
    :param steps_in_loop:
    :param rules_to_store_f:
    :return:
    """
    one_step = step.get_timestep(
        all_params=all_params, stuff_for_time_loop=stuff_for_time_loop
    )

    if all_params["backend"]["core"] == "numpy":
        import numpy as np

        def inner_loop(time_array, driver_array, temp_storage):
            t0 = time()
            temp_storage["time_batch"] = time_array
            temp_storage["driver_array_batch"] = driver_array
            for it in tqdm(range(steps_in_loop)):
                temp_storage, _ = one_step(temp_storage, it)
            post_inner_loop_update(temp_storage, t0, np)

            return temp_storage

        return inner_loop

    else:
        raise NotImplementedError
