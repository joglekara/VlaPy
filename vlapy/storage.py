import xarray as xr
import numpy as np
import os


class StorageManager:
    def __init__(self, xax, vax, tax, base_path):
        efield_path = os.path.join(base_path, "electric_field_vs_time.nc")
        self.efield_arr = self.__init_electric_field_storage(tax, xax, efield_path)

        f_path = os.path.join(base_path, "dist_func_vs_time.nc")
        self.f_arr = self.__init_dist_func_storage(tax, xax, vax, f_path)

    def __init_electric_field_storage(self, tax, xax, path):
        """
        Initialize electric field storage DataArray

        :param tax:
        :param xax:
        :param path:
        :return:
        """
        electric_field_store = np.zeros((tax.size, xax.size))

        ef_DA = xr.DataArray(
            data=electric_field_store, coords=[tax, xax], dims={"t", "x"},
        )

        ef_DA.to_netcdf(path)

        return xr.load_dataarray(path)

    def __init_dist_func_storage(self, tax, xax, vax, path):
        """
        Initialize distribution function storage

        :param tax:
        :param xax:
        :param vax:
        :param path:
        :return:
        """
        dist_func_store = np.zeros((tax.size, xax.size, vax.size))

        f_DA = xr.DataArray(
            data=dist_func_store, coords=[tax, xax, vax], dims={"t", "x", "v"},
        )

        f_DA.to_netcdf(path)

        return xr.load_dataarray(path)

    def batched_write_to_file(self, t_range, e, f):
        """
        Write batched to file

        This is to save time by keeping some of the history on
        accelerator rather than passing it back every time step


        :param t_range:
        :param e:
        :param f:
        :return:
        """
        self.efield_arr.loc[t_range, :] = e
        self.f_arr.loc[t_range, :] = f
