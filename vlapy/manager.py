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


import tempfile

from tqdm import tqdm
from time import time
import mlflow
import numpy as np

from vlapy import storage, outer_loop
from vlapy.infrastructure import print_to_screen


def start_run(all_params, pulse_dictionary, diagnostics, uris, name="test"):
    """
    This is the highest level function that calls the time integration loop

    MLFlow is initialized here.
    Domain configuration is also performed here.
    All file storage is initialized here.

    :param all_params: (dictionary) contains the input parameters of the simulation
    :param pulse_dictionary: (dictionary) contains the parameters for the ponderomotive force driver
    :param diagnostics: (vlapy.Diagnostics) contains the diagnostics routine for this particular simulation
    :param uris: (string) the location of the mlflow server
    :param name: (string) the name of the MLFlow experiment
    :return: (Mlflow.Run) returns the completed Run object
    """

    t0 = time()
    print_to_screen.print_startup_message(name, all_params, pulse_dictionary, uris)
    if "local" not in uris["tracking"].casefold():
        mlflow.set_tracking_uri(uris["tracking"])

    exp = mlflow.set_experiment(name)

    with mlflow.start_run(experiment_id=exp.experiment_id) as run:
        with tempfile.TemporaryDirectory() as temp_path:

            if diagnostics.rules_to_store_f["space"] == "all":
                x_storage_multiplier = all_params["nx"]
            elif diagnostics.rules_to_store_f["space"][0] == "k0":
                x_storage_multiplier = 2 * len(diagnostics.rules_to_store_f["space"])

            mem_f_store = x_storage_multiplier * all_params["nv"]
            mem_field_store = all_params["nx"] * 8

            # Initialize loop parameters
            steps_in_loop = int(
                1e9
                * all_params["backend"]["max_GB_for_device"]
                / (6 * (mem_f_store + mem_field_store) * 8)
            )
            if steps_in_loop > all_params["nt"]:
                steps_in_loop = int(all_params["nt"] / 1.25)

            n_loops = all_params["nt"] // steps_in_loop + 1
            actual_num_steps = n_loops * steps_in_loop

            all_params["steps_in_loop"] = steps_in_loop
            all_params["n_loops"] = n_loops
            all_params["actual_num_steps"] = actual_num_steps

            # Get numpy arrays of the simulation configuration
            stuff_for_time_loop = outer_loop.get_everything_ready_for_outer_loop(
                diagnostics=diagnostics,
                all_params=all_params,
                pulse_dictionary=pulse_dictionary,
                overall_num_steps=actual_num_steps,
            )

            # Initialize the storage manager -- folders, parameters, etc.
            storage_manager = storage.StorageManager(
                xax=stuff_for_time_loop["x"],
                vax=stuff_for_time_loop["v"],
                f=stuff_for_time_loop["f"],
                base_path=temp_path,
                rules_to_store_f=diagnostics.rules_to_store_f,
                num_steps_in_one_loop=steps_in_loop,
                all_params=all_params,
                pulse_dictionary=pulse_dictionary,
            )

            mlflow.log_metrics(metrics={"startup_time": time() - t0}, step=0)
            t0 = time()

            sim_config, do_inner_loop = outer_loop.get_sim_config_and_inner_loop_step(
                all_params=all_params,
                stuff_for_time_loop=stuff_for_time_loop,
                nt_in_loop=steps_in_loop,
                store_f_rules=diagnostics.rules_to_store_f,
            )

            mlflow.log_metrics(metrics={"compile_time": time() - t0}, step=0)
            t0 = time()

            # TODO: Could support resume here
            it_start = 0

            # We run a higher level loop and a lower level loop.
            # The higher level loop contains:
            # 1 - Get the driver for all time-steps involved in this iteration of the lower level loop
            # 2 - Perform the lower level loop
            # 3 - Write the output to file
            #
            # The lower level loop is a loop over `steps_in_loop` timesteps. It is entirely executed on the
            # accelerator. The size of this loop can be controlled by the `MAX_DOUBLES_IN_FILE` parameter.
            # The goal was to allow that parameter to control the amount of memory needed on the accelerator
            for it in tqdm(range(it_start, n_loops * steps_in_loop, steps_in_loop)):
                curr_time_index = np.arange(it, it + steps_in_loop)

                # Get driver and time array for the duration of the lower level loop
                driver_array = np.array(stuff_for_time_loop["driver"][curr_time_index])
                time_array = np.array(stuff_for_time_loop["t"][curr_time_index])

                # Perform lower level loop
                sim_config = do_inner_loop(
                    temp_storage=sim_config,
                    driver_array=driver_array,
                    time_array=time_array,
                )

                mlflow.log_metrics(
                    metrics={"calculation_time": (time() - t0) / steps_in_loop}, step=it
                )
                t0 = time()

                # Perform a batched data update with the lower level loop output
                storage_manager.batch_update(sim_config)

                mlflow.log_metrics(metrics={"batch_update_time": time() - t0}, step=it)
                t0 = time()

                # Run the diagnostics on the simulation so far
                diagnostics(storage_manager)

                mlflow.log_metrics(metrics={"diagnostic_time": time() - t0}, step=it)
                mlflow.log_metrics(
                    metrics={
                        "diagnostic_time_averaged_over_sim_time": (time() - t0)
                        / ((it + 1) * steps_in_loop)
                    },
                    step=it,
                )
                t0 = time()

                # Log the artifacts
                storage_manager.log_artifacts()

                mlflow.log_metrics(metrics={"logging_time": time() - t0}, step=it)
                t0 = time()

            storage_manager.close()
            del storage_manager

    return run
