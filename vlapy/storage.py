import xarray as xr
import numpy as np
import os


class StorageManager:
    def __init__(self, xax, vax, tax, base_path):
        self.base_path = base_path
        self.efield_path = os.path.join(base_path, "electric_field_vs_time.nc")
        self.f_path = os.path.join(base_path, "dist_func_vs_time.nc")

        self.__init_electric_field_storage(tax=tax, xax=xax)
        self.__init_dist_func_storage(tax=tax, xax=xax, vax=vax)

    def close(self):
        self.efield_arr.to_netcdf(self.efield_path)
        self.f_arr.to_netcdf(self.f_path)

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

        ef_DA.to_netcdf(self.efield_path)

        self.efield_arr = xr.open_dataarray(self.efield_path)

    def __init_dist_func_storage(self, tax, xax, vax):
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
            data=dist_func_store,
            coords=[("time", tax), ("space", xax), ("velocity", vax)],
        )

        f_DA.to_netcdf(self.f_path)

        self.f_arr = xr.open_dataarray(self.f_path)

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
        t_xr = xr.DataArray(data=t_range, dims=["time"])

        self.efield_arr.loc[t_xr, :] = e
        self.f_arr.loc[t_xr, :] = f

        # Save and reopen
        self.efield_arr.to_netcdf(self.efield_path)
        self.f_arr.to_netcdf(self.f_path)

        self.efield_arr = xr.open_dataarray(self.efield_path)
        self.f_arr = xr.open_dataarray(self.f_path)
