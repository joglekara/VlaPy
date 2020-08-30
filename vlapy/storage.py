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
        individual_path,
        combine="by_coords",
        engine="h5netcdf",
        parallel=True,
    )

    arr.to_netcdf(
        overall_path,
        engine="h5netcdf",
        invalid_netcdf=True,
    )

    arr = xr.open_dataset(
        overall_path,
        engine="h5netcdf",
    )

    return arr


def get_batched_data_from_sim_config(sim_config):
    f = sim_config["stored_f"]
    current_f = sim_config["f"]
    time_batch = sim_config["time_batch"]
    series = sim_config["series"]
    fields = sim_config["fields"]

    return fields, f, time_batch, series, current_f


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
        "series": os.path.join(base_path, "long_term", "series"),
        "fields": os.path.join(base_path, "long_term", "fields"),
        "distribution": os.path.join(base_path, "long_term", "distribution_function"),
        "series-individual": os.path.join(base_path, "temp", "series"),
        "fields-individual": os.path.join(base_path, "temp", "fields"),
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
        f,
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
        self.init_f = f
        self.num_timesteps_to_store = num_steps_in_one_loop

        self.write_parameters_to_file(all_params, "all_parameters")
        self.write_parameters_to_file(pulse_dictionary, "pulses")

    def close(self):
        """
        This method removes the temporary directories and files

        :return:
        """

        shutil.rmtree(self.paths["distribution-individual"])
        shutil.rmtree(self.paths["fields-individual"])

    def batch_update(self, sim_config):
        """
        This method updates the storage arrays by batch by fetching them from the accelerator

        :param sim_config:
        :return:
        """

        dict_of_stored_dists = {}

        (
            dict_of_stored_fields,
            dict_of_stored_dists["distribution_function"],
            time_actually_stored,
            series,
            self.current_f,
        ) = get_batched_data_from_sim_config(sim_config)

        self.write_series_batch(
            time_actually_stored=time_actually_stored,
            dict_of_stored_series=series,
        )
        self.write_field_batch(
            time_actually_stored=time_actually_stored,
            dict_of_stored_fields=dict_of_stored_fields,
        )
        self.write_dist_batch(
            time_actually_stored=time_actually_stored,
            dict_of_stored_dist=dict_of_stored_dists,
        )
        self.stored += 1

    def write_field_batch(self, time_actually_stored, dict_of_stored_fields):
        """
        This writes a batch of 1D arrays to file

        :param time_actually_stored:
        :param dict_of_stored_fields:
        :return:
        """

        field_coords = [("time", time_actually_stored), ("space", self.xax)]
        self.__write_batch__(
            dict_of_stored_fields,
            coords=field_coords,
            individual_file_name=os.path.join(
                self.paths["fields-individual"], format(self.stored, "03") + ".nc"
            ),
        )

    def write_series_batch(self, time_actually_stored, dict_of_stored_series):
        """
        This writes a batch of 1D arrays to file

        :param time_actually_stored:
        :param dict_of_stored_fields:
        :return:
        """

        series_coords = [("time", time_actually_stored)]
        self.__write_batch__(
            dict_of_stored_series,
            coords=series_coords,
            individual_file_name=os.path.join(
                self.paths["series-individual"], format(self.stored, "03") + ".nc"
            ),
        )

    def write_dist_batch(self, time_actually_stored, dict_of_stored_dist):

        if not isinstance(self.rules_to_store_f, dict):
            raise NotImplementedError

        f_coords = [
            ("time", time_actually_stored),
            (None, None),
            ("velocity", self.vax),
        ]

        if self.rules_to_store_f["space"] == "all":
            f_coords[1] = ("space", self.xax)

        elif self.rules_to_store_f["space"][0] == "k0":
            f_coords[1] = (
                "fourier_mode",
                np.linspace(
                    0,
                    len(self.rules_to_store_f["space"]) - 1,
                    len(self.rules_to_store_f["space"]),
                ),
            )
        else:
            raise NotImplementedError

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

        self.__write_batch__(
            dict_of_stored_dist,
            coords=f_coords,
            individual_file_name=os.path.join(
                self.paths["distribution-individual"],
                format(self.f_store_index, "03") + ".nc",
            ),
        )

    def __write_batch__(self, dict_of_stored_data, coords, individual_file_name):
        """
        This writes a batch of 1D arrays to file

        :param time_actually_stored:
        :param dict_of_stored_fields:
        :return:
        """

        dataarray_dict = {}

        for name, array in dict_of_stored_data.items():
            dataarray_dict[name] = xr.DataArray(
                data=array,
                coords=coords,
            )

        ds = xr.Dataset(data_vars=dataarray_dict)

        # Save
        ds.to_netcdf(individual_file_name, engine="h5netcdf")
        del ds

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

        self.series_dataset = load_over_all_timesteps(
            individual_path=self.paths["series-individual"] + "/*.nc",
            overall_path=os.path.join(self.paths["series"], "all-series.nc"),
        )

        self.fields_dataset = load_over_all_timesteps(
            individual_path=self.paths["fields-individual"] + "/*.nc",
            overall_path=os.path.join(self.paths["fields"], "all-fields.nc"),
        )

        self.dist_dataset = load_over_all_timesteps(
            individual_path=self.paths["distribution-individual"] + "/*.nc",
            overall_path=os.path.join(
                self.paths["distribution"], "all-distribution.nc"
            ),
        )

    def log_artifacts(self):
        mlflow.log_artifacts(self.paths["long_term"])

    def unload_data_over_all_timesteps(self):
        del self.series_dataset
        del self.fields_dataset
        del self.dist_dataset
