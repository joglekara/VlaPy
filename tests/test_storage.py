from vlapy import storage
import numpy as np
import os
import uuid
import shutil


def test_storage_init_files_exist():
    xax = np.linspace(0, 1, 16)
    vax = np.linspace(0, 1, 24)
    tax = np.linspace(0, 1, 32)

    dirname = os.path.join(os.getcwd(), str(uuid.uuid4()))
    os.makedirs(dirname, exist_ok=True)

    st = storage.StorageManager(xax=xax, vax=vax, tax=tax, base_path=dirname)

    assert os.path.exists(os.path.join(dirname, "electric_field_vs_time.nc"))
    assert os.path.exists(os.path.join(dirname, "dist_func_vs_time.nc"))

    shutil.rmtree(dirname)


def test_storage_init_shape():
    xax = np.linspace(0, 1, 16)
    vax = np.linspace(0, 1, 24)
    tax = np.linspace(0, 1, 32)

    dirname = os.path.join(os.getcwd(), str(uuid.uuid4()))
    os.makedirs(dirname, exist_ok=True)

    st = storage.StorageManager(xax=xax, vax=vax, tax=tax, base_path=dirname)

    np.testing.assert_equal(st.efield_arr.coords["space"].size, xax.size)
    np.testing.assert_equal(st.efield_arr.coords["time"].size, tax.size)

    np.testing.assert_equal(st.f_arr.coords["space"].size, xax.size)
    np.testing.assert_equal(st.f_arr.coords["time"].size, tax.size)
    np.testing.assert_equal(st.f_arr.coords["velocity"].size, vax.size)

    shutil.rmtree(dirname)
