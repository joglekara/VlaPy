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

import xarray as xr
import numpy as np


class StorageManager:
    def __init__(self, xax, vax, tax, base_path, store_f=None):
        """
        This is the initialization for the storage class.
        The paths are set

        At the moment, we store the electric field at every timestep
        we store the driver field at every timestep
        and we have the choice of storing the distribution function at every time-step
        but since it's typically too large, we can store a few choice Fourier modes

        :param xax:
        :param vax:
        :param tax:
        :param base_path:
        :param store_f:
        """
        self.initialize_temporary_storage(xax, vax, tax, store_f)

        self.base_path = base_path
        self.efield_path = os.path.join(base_path, "electric_field_vs_time.nc")
        self.driver_efield_path = os.path.join(
            base_path, "driver_electric_field_vs_time.nc"
        )
        self.f_path = os.path.join(base_path, "dist_func_vs_time.nc")

        self.__init_electric_field_storage(tax=tax, xax=xax)

        self.store_f = store_f

        if store_f is not None:
            self.__init_dist_func_storage(
                tax=tax, xax=xax, vax=vax, f_storage_rules=store_f
            )

    def close(self):
        """
        This method closes all the files safely before program termination

        :return:
        """
        self.efield_arr.to_netcdf(
            self.efield_path, engine="h5netcdf", invalid_netcdf=True
        )
        self.driver_efield_arr.to_netcdf(
            self.driver_efield_path, engine="h5netcdf", invalid_netcdf=True
        )
        if self.store_f:
            self.f_arr.to_netcdf(self.f_path, engine="h5netcdf", invalid_netcdf=True)

    def initialize_temporary_storage(self, xax, vax, tax, store_f):
        """
        This method initializes storage

        :param xax:
        :param vax:
        :param tax:
        :param store_f:
        :return:
        """
        nt = tax.size
        nx = xax.size
        nv = vax.size

        if nt // 4 < 100:
            self.t_store = 100
        else:
            self.t_store = nt // 4

        self.temp_field_store = np.zeros([self.t_store, nx])
        self.temp_driver_store = np.zeros([self.t_store, nx])
        self.temp_t_store = np.zeros(self.t_store)
        self.it_store = 0

        if store_f == "all-x":
            self.temp_dist_store = np.zeros([self.t_store, nx, nv])
        else:
            kax = np.linspace(0, 1, 2)
            self.temp_dist_store = np.zeros(
                (self.t_store, kax.size, nv), dtype=np.complex
            )

    def __init_electric_field_storage(self, tax, xax):
        """
        Initialize electric field storage DataArray

        :param tax:
        :param xax:
        :param path:
        :return:
        """

        electric_field_store = np.zeros((tax.size, xax.size))

        ef_DA = xr.DataArray(
            data=electric_field_store, coords=[("time", tax), ("space", xax)]
        )

        ef_DA.to_netcdf(self.efield_path, engine="h5netcdf", invalid_netcdf=True)

        self.efield_arr = xr.load_dataarray(self.efield_path, engine="h5netcdf")

        driver_electric_field_store = np.zeros((tax.size, xax.size))

        driver_ef_DA = xr.DataArray(
            data=driver_electric_field_store, coords=[("time", tax), ("space", xax)]
        )

        driver_ef_DA.to_netcdf(
            self.driver_efield_path, engine="h5netcdf", invalid_netcdf=True
        )

        self.driver_efield_arr = xr.load_dataarray(
            self.driver_efield_path, engine="h5netcdf"
        )

    def __init_dist_func_storage(self, tax, xax, vax, f_storage_rules):
        """
        Initialize distribution function storage

        :param tax:
        :param xax:
        :param vax:
        :param path:
        :return:
        """

        if f_storage_rules == "all-x":
            dist_func_store = np.zeros((tax.size, xax.size, vax.size))
            f_DA = xr.DataArray(
                data=dist_func_store,
                coords=[("time", tax), ("space", xax), ("velocity", vax)],
            )
        else:
            kax = np.linspace(0, 1, 2)
            dist_func_store = np.zeros((tax.size, kax.size, vax.size), dtype=np.complex)
            f_DA = xr.DataArray(
                data=dist_func_store,
                coords=[("time", tax), ("fourier_mode", kax), ("velocity", vax)],
            )

        f_DA.to_netcdf(self.f_path, engine="h5netcdf", invalid_netcdf=True)

        self.f_arr = xr.load_dataarray(self.f_path, engine="h5netcdf")

    def temp_update(self, tt, f, e, driver):
        """
        This is the method that performs an update at the end of every time step

        :param tt:
        :param f:
        :param e:
        :param driver:
        :return:
        """
        self.temp_t_store[self.it_store] = tt
        self.temp_field_store[self.it_store] = e
        self.temp_driver_store[self.it_store] = driver

        if self.store_f is not None:
            if self.store_f == "all-x":
                self.temp_dist_store[self.it_store] = f
            else:
                fk = np.fft.fft(f, axis=0, norm="ortho")
                self.temp_dist_store[self.it_store, 0, :] = fk[
                    0,
                ]
                self.temp_dist_store[self.it_store, 1, :] = fk[
                    1,
                ]

        self.temp_ticker_update()

    def temp_ticker_update(self):
        """
        This is the method that updates the storage ticker at every timestep
        It also checks so to see if this is the timestep to write to file

        :return:
        """

        self.it_store += 1

        if self.it_store == self.t_store:
            self.batched_write_to_file()
            self.it_store = 0

    def batched_write_to_file(self):
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
        t_xr = xr.DataArray(data=self.temp_t_store, dims=["time"])

        self.efield_arr.loc[t_xr, :] = self.temp_field_store
        self.driver_efield_arr.loc[t_xr, :] = self.temp_driver_store

        # Save and reopen
        self.efield_arr.to_netcdf(
            self.efield_path, engine="h5netcdf", invalid_netcdf=True
        )
        self.driver_efield_arr.to_netcdf(
            self.driver_efield_path, engine="h5netcdf", invalid_netcdf=True
        )

        self.efield_arr = xr.load_dataarray(self.efield_path, engine="h5netcdf")
        self.driver_efield_arr = xr.load_dataarray(
            self.driver_efield_path, engine="h5netcdf"
        )

        if self.store_f is not None:
            self.f_arr.loc[t_xr,] = self.temp_dist_store
            self.f_arr.to_netcdf(self.f_path, engine="h5netcdf", invalid_netcdf=True)
            self.f_arr = xr.load_dataarray(self.f_path, engine="h5netcdf")

    def write_parameters_to_file(self, param_dict, filename):
        """
        This is a helper function in case anything else needs to be written to file

        :param param_dict:
        :param filename:
        :return:
        """
        with open(os.path.join(self.base_path, filename + ".txt"), "w") as fi:
            json.dump(param_dict, fi)
