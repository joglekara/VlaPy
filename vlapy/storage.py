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

import os
import json
import shutil

import mlflow
import xarray as xr
import numpy as np


def load_over_all_timesteps(individual_path, overall_path):
    """
    This function
    1 - loads multiple dataarrays
    2 - concatenates them into one dataset
    3 - writes to file

    :param individual_path:
    :param overall_path:
    :return:
    """
    arr = xr.open_mfdataset(
        individual_path, combine="by_coords", engine="h5netcdf", parallel=True,
    )

    arr.to_netcdf(
        overall_path, engine="h5netcdf", invalid_netcdf=True,
    )

    arr = xr.open_dataset(overall_path, engine="h5netcdf",)

    return arr


def get_batched_data_from_sim_config(sim_config):
    e = sim_config["stored_e"]
    f = sim_config["stored_f"]
    time_batch = sim_config["time_batch"]
    driver_batch = sim_config["driver_array_batch"]
    health = sim_config["health"]

    return e, f, time_batch, driver_batch, health


def get_paths(base_path):
    """
    This function writes the paths to a dictionary and also makes the folder structure

    :param base_path:
    :return:
    """
    paths = {
        "base": base_path,
        "long_term": os.path.join(base_path, "long_term"),
        "temp": os.path.join(base_path, "temp"),
        "e": os.path.join(base_path, "long_term", "electric_field"),
        "driver": os.path.join(base_path, "long_term", "driver_electric_field"),
        "distribution": os.path.join(base_path, "long_term", "distribution_function"),
        "e-individual": os.path.join(base_path, "temp", "electric_field"),
        "driver-individual": os.path.join(base_path, "temp", "driver_electric_field"),
        "distribution-individual": os.path.join(
            base_path, "temp", "distribution_function"
        ),
    }

    for key, val in paths.items():
        os.makedirs(val, exist_ok=True)

    return paths


class StorageManager:
    def __init__(
        self,
        xax,
        vax,
        base_path,
        num_steps_in_one_loop,
        all_params,
        pulse_dictionary,
        rules_to_store_f=None,
    ):
        """
        This is the initialization for the storage class.
        The paths are set

        At the moment, we store the electric field at every timestep
        we store the driver field at every timestep
        and we have the choice of storing the distribution function at every time-step
        but since it's typically too large, we can store a few choice Fourier modes

        e.g. if `store_f = ["k0","k1","k2"]`, VlaPy will store the 0th and first 2 real-space Fourier modes
        of the system

        :param x: real-space axis (numpy array of shape (nx,))
        :param v: velocity axis (numpy array of shape (nv,))
        :param t: time axis (numpy array of shape (nt,))
        :param base_path: path to run folder (string)
        :param rules_to_store_f: list of Fourier modes to store (len = 0+`num_fourier_modes`)
        """

        self.paths = get_paths(base_path)
        self.rules_to_store_f = rules_to_store_f
        self.stored = 0
        self.xax = xax
        self.vax = vax
        self.num_timesteps_to_store = num_steps_in_one_loop

        self.initialize_storage()

        self.write_parameters_to_file(all_params, "all_parameters")
        self.write_parameters_to_file(pulse_dictionary, "pulses")

    def close(self):
        """
        This method closes all the files safely before program termination

        :return:
        """

        del self.field_store
        del self.dist_store
        del self.time_store
        del self.driver_store

        shutil.rmtree(self.paths["distribution-individual"])
        shutil.rmtree(self.paths["e-individual"])
        shutil.rmtree(self.paths["driver-individual"])

    def initialize_storage(self):
        """
        This method initializes storage

        :param x: real-space axis (numpy array of shape (nx,))
        :param v: velocity axis (numpy array of shape (nv,))
        :param t: time axis (numpy array of shape (nt,))
        :param rules_to_store_f: list of Fourier modes to store (len = 0+`num_fourier_modes`)
        :return:
        """
        nx = self.xax.size
        nv = self.vax.size

        self.field_store = np.zeros([self.num_timesteps_to_store, nx])
        self.driver_store = np.zeros([self.num_timesteps_to_store, nx])
        self.time_store = np.zeros(self.num_timesteps_to_store)

        if self.rules_to_store_f["space"] == "all-x":
            self.dist_store = np.zeros([self.num_timesteps_to_store, nx, nv])
        elif self.rules_to_store_f["space"][0] == "k0":
            kax = np.linspace(
                0,
                len(self.rules_to_store_f["space"]) - 1,
                len(self.rules_to_store_f["space"]),
            )
            self.dist_store = np.zeros(
                (self.num_timesteps_to_store, kax.size, nv), dtype=np.complex
            )
        else:
            raise NotImplementedError

        self.health = {}

        self.stored_quantities = ["e", "driver", "distribution"]
        self.overall_arrs = {}

    def batch_update(self, sim_config):
        """
        This method updates the storage arrays by batch by fetching them from the accelerator

        :return:
        """

        e, f, current_time, driver, health = get_batched_data_from_sim_config(
            sim_config
        )

        self.time_store = current_time
        self.field_store = e
        self.driver_store = driver
        self.dist_store = f
        self.health = health

        self.__batched_write_to_file__()
        self.stored += 1

    def write_f_batch(self):
        """
        This function writes a batch of the distribution function

        :return:
        """
        if self.rules_to_store_f is None:
            pass
        else:
            f_coords = [
                ("time", self.time_actually_stored),
                (None, None),
                ("velocity", self.vax),
            ]

            if self.rules_to_store_f["space"] == "all-x":
                f_coords[1] = ("space", self.xax)

            elif self.rules_to_store_f["space"][0] == "k0":
                f_coords[1] = (
                    "fourier_mode",
                    np.linspace(
                        0, len(self.rules_to_store_f) - 1, len(self.rules_to_store_f)
                    ),
                )

            else:
                raise NotImplementedError

            f_arr = xr.DataArray(
                data=self.dist_store[: self.time_store_actual_end,], coords=f_coords,
            )

            f_arr_ds = f_arr.to_dataset(name="distribution_function")

            if self.rules_to_store_f["time"] == "first-last":
                if self.stored == 0:
                    self.f_store_index = 0
                else:
                    self.f_store_index = 999
            elif self.rules_to_store_f["time"] == "all":
                if self.stored == 0:
                    self.f_store_index = 0
                else:
                    self.f_store_index += 1
            else:
                raise NotImplementedError

            f_arr_ds.to_netcdf(
                os.path.join(
                    self.paths["distribution-individual"],
                    format(self.f_store_index, "03") + ".nc",
                ),
                engine="h5netcdf",
                invalid_netcdf=True,
            )

            del f_arr_ds

    def write_field_batch(self, dataset_name, individual_filepath):
        """
        This writes a batch of 1D arrays to file

        :param dataset_name:
        :param individual_filepath:
        :return:
        """

        field_coords = [("time", self.time_actually_stored), ("space", self.xax)]

        field_arr = xr.DataArray(
            data=self.field_store[: self.time_store_actual_end,], coords=field_coords,
        )
        field_ds = field_arr.to_dataset(name=dataset_name)
        # Save
        field_ds.to_netcdf(
            os.path.join(individual_filepath, format(self.stored, "03") + ".nc"),
            engine="h5netcdf",
        )
        del field_ds

    def __batched_write_to_file__(self):
        """
        Write batched to file

        This is to save time by keeping some of the history on
        accelerator rather than passing it back every time step


        :param t_range:
        :param e:
        :param e_driver:
        :param f:
        :return:
        """

        self.time_store_actual_end = self.time_store.size
        self.time_actually_stored = self.time_store

        for ds, pth in [("driver", "driver"), ("electric", "e")]:
            self.write_field_batch(
                dataset_name=ds + "_field",
                individual_filepath=self.paths[pth + "-individual"],
            )

        self.write_f_batch()

    def write_parameters_to_file(self, param_dict, filename):
        """
        This is a helper function in case anything else needs to be written to file

        :param param_dict:
        :param filename:
        :return:
        """
        with open(os.path.join(self.paths["base"], filename + ".txt"), "w") as fi:
            json.dump(param_dict, fi)

    def load_data_over_all_timesteps(self):
        """

        :return:
        """
        self.overall_arrs = {}
        for quantity in self.stored_quantities:
            self.overall_arrs[quantity] = load_over_all_timesteps(
                individual_path=self.paths[quantity + "-individual"] + "/*.nc",
                overall_path=os.path.join(
                    self.paths[quantity], "all-" + quantity + ".nc"
                ),
            )

    def __delete_distribution__(self):
        shutil.rmtree(os.path.join(self.paths["distribution"]))

    def log_artifacts(self):
        mlflow.log_artifacts(self.paths["long_term"])

    def unload_data_over_all_timesteps(self):
        del self.overall_arrs
