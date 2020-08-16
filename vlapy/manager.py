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
import mlflow
import numpy as np

from vlapy import initializers, storage, inner_loop


def start_run(all_params, pulse_dictionary, diagnostics, uris, name="test"):
    """
    This is the highest level function that calls the time integration loop

    MLFlow is initialized here.
    Domain configuration is also performed here.
    All file storage is initialized here.

    :param all_params:
    :param pulse_dictionary:
    :param diagnostics:
    :param name:
    :param mlflow_path:
    :return:
    """

    if "local" not in uris["tracking"].casefold():
        mlflow.set_tracking_uri(uris["tracking"])

    exp_id = mlflow.set_experiment(name)

    with mlflow.start_run(experiment_id=exp_id) as run:
        with tempfile.TemporaryDirectory() as temp_path:
            # Initialize loop parameters
            steps_in_loop = int(all_params["backend"]["max_doubles_per_file"]) // (
                all_params["nx"] * all_params["nv"]
            )
            n_loops = all_params["nt"] // steps_in_loop + 1
            actual_num_steps = n_loops * steps_in_loop

            all_params["steps_in_loop"] = steps_in_loop
            all_params["n_loops"] = n_loops
            all_params["actual_num_steps"] = actual_num_steps

            # Get numpy arrays of the simulation configuration
            stuff_for_time_loop = initializers.get_everything_ready_for_time_loop(
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

            sim_config, do_inner_loop = inner_loop.get_inner_loop(
                all_params=all_params,
                stuff_for_time_loop=stuff_for_time_loop,
                steps_in_loop=steps_in_loop,
                rules_to_store_f=diagnostics.rules_to_store_f,
            )

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
                curr_time_index = range(it, it + steps_in_loop)

                # Get driver and time array for the duration of the lower level loop
                driver_array = np.array(stuff_for_time_loop["driver"][curr_time_index])
                time_array = np.array(stuff_for_time_loop["t"][curr_time_index])

                # Perform lower level loop
                sim_config = do_inner_loop(
                    temp_storage=sim_config,
                    driver_array=driver_array,
                    time_array=time_array,
                )

                # Perform a batched data update with the lower level loop output
                storage_manager.batch_update(sim_config)

                # Run the diagnostics on the simulation so far
                diagnostics(storage_manager)

                # Log the artifacts
                storage_manager.log_artifacts()

            storage_manager.close()
            del storage_manager

    return run
