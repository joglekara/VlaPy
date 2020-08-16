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
from time import time
from tqdm import tqdm

from vlapy.core import step


def get_arrays_for_time_loop(stuff_for_time_loop, nt_in_loop, store_f_rules):
    """
    This function converts the previously created NumPy arrays to JAX arrays.

    It also creates the temporary storage for e and f that is used for the
    low-level time-loop

    :param stuff_for_time_loop:
    :param nt_in_loop:
    :param store_f_rules:
    :return:
    """

    # This is where we initialize the right distribution function storage array
    # If we're saving all the "x" values then that array is created
    if store_f_rules["space"] == "all":
        store_f = np.zeros(
            (nt_in_loop,) + stuff_for_time_loop["f"].shape, dtype=np.complex64
        )
        store_f[0,] = stuff_for_time_loop["f"]

    # If we're saving the first M spatial-modes then that array is created
    elif isinstance(store_f_rules["space"], list) and store_f_rules["space"][0] == "k0":
        store_f = np.zeros(
            (nt_in_loop, len(store_f_rules), stuff_for_time_loop["f"].shape[1]),
            dtype=np.complex64,
        )

        store_f[0,] = np.fft.fft(stuff_for_time_loop["f"], axis=0)[: len(store_f_rules)]

    else:
        raise NotImplementedError

    # This is where we return the whole simulation configuration dictionary
    return {
        "time_batch": np.zeros(nt_in_loop),
        "e": np.array(stuff_for_time_loop["e"]),
        "f": np.array(stuff_for_time_loop["f"]),
        "stored_f": np.array(store_f),
        "health": {
            "mean_n": np.zeros(nt_in_loop),
            "mean_v": np.zeros(nt_in_loop),
            "mean_T": np.zeros(nt_in_loop),
            "mean_e2": np.zeros(nt_in_loop),
            "mean_de2": np.zeros(nt_in_loop),
            "mean_t_plus_e2_minus_de2": np.zeros(nt_in_loop),
            "mean_t_plus_e2_plus_de2": np.zeros(nt_in_loop),
            "mean_f2": np.zeros(nt_in_loop),
            "mean_flogf": np.zeros(nt_in_loop),
        },
        "fields": {
            "e": np.zeros((nt_in_loop,) + stuff_for_time_loop["e"].shape),
            "driver": np.zeros((nt_in_loop,) + stuff_for_time_loop["e"].shape),
            "n": np.zeros((nt_in_loop,) + stuff_for_time_loop["e"].shape),
            "j": np.zeros((nt_in_loop,) + stuff_for_time_loop["e"].shape),
            "T": np.zeros((nt_in_loop,) + stuff_for_time_loop["e"].shape),
            "q": np.zeros((nt_in_loop,) + stuff_for_time_loop["e"].shape),
            "fv4": np.zeros((nt_in_loop,) + stuff_for_time_loop["e"].shape),
            "vN": np.zeros((nt_in_loop,) + stuff_for_time_loop["e"].shape),
        },
    }


def get_numpy_inner_loop_stepper(
    all_params, stuff_for_time_loop, steps_in_loop, rules_to_store_f
):
    """
    This is where the `NumPy`-based inner-loop is created and returned

    :param all_params:
    :param stuff_for_time_loop:
    :param steps_in_loop:
    :param rules_to_store_f:
    :return:
    """
    one_numpy_step = step.get_timestep(
        all_params=all_params, stuff_for_time_loop=stuff_for_time_loop
    )
    sim_config = get_arrays_for_time_loop(
        stuff_for_time_loop, steps_in_loop, rules_to_store_f
    )

    def numpy_inner_loop(time_array, driver_array, temp_storage):
        t0 = time()

        temp_storage["time_batch"] = time_array
        temp_storage["driver_array_batch"] = driver_array
        for it in tqdm(range(steps_in_loop)):
            temp_storage, _ = one_numpy_step(temp_storage, it)

        temp_storage["time_for_batch"] = time() - t0

        return temp_storage

    return sim_config, numpy_inner_loop


def get_inner_loop(
    all_params, stuff_for_time_loop, steps_in_loop, rules_to_store_f,
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
        sim_config, do_inner_loop = get_numpy_inner_loop_stepper(
            all_params, stuff_for_time_loop, steps_in_loop, rules_to_store_f
        )
    else:
        raise NotImplementedError

    return sim_config, do_inner_loop
