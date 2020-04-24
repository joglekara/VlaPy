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
import uuid
import shutil

import numpy as np

from vlapy import storage


def test_storage_init_files_exist():
    xax = np.linspace(0, 1, 16)
    vax = np.linspace(0, 1, 24)
    tax = np.linspace(0, 1, 32)

    dirname = os.path.join(os.getcwd(), str(uuid.uuid4()))
    os.makedirs(dirname, exist_ok=True)

    st = storage.StorageManager(
        xax=xax, vax=vax, tax=tax, base_path=dirname, store_f="all-x"
    )

    assert os.path.exists(os.path.join(dirname, "electric_field_vs_time.nc"))
    assert os.path.exists(os.path.join(dirname, "dist_func_vs_time.nc"))

    shutil.rmtree(dirname)


def test_storage_init_shape():
    xax = np.linspace(0, 1, 16)
    vax = np.linspace(0, 1, 24)
    tax = np.linspace(0, 1, 32)

    dirname = os.path.join(os.getcwd(), str(uuid.uuid4()))
    os.makedirs(dirname, exist_ok=True)

    st = storage.StorageManager(
        xax=xax, vax=vax, tax=tax, base_path=dirname, store_f="all-x"
    )

    np.testing.assert_equal(st.efield_arr.coords["space"].size, xax.size)
    np.testing.assert_equal(st.efield_arr.coords["time"].size, tax.size)

    np.testing.assert_equal(st.f_arr.coords["space"].size, xax.size)
    np.testing.assert_equal(st.f_arr.coords["time"].size, tax.size)
    np.testing.assert_equal(st.f_arr.coords["velocity"].size, vax.size)

    shutil.rmtree(dirname)


def test_storage_init_shape_fourier():
    xax = np.linspace(0, 1, 16)
    vax = np.linspace(0, 1, 24)
    tax = np.linspace(0, 1, 32)

    dirname = os.path.join(os.getcwd(), str(uuid.uuid4()))
    os.makedirs(dirname, exist_ok=True)

    st = storage.StorageManager(
        xax=xax, vax=vax, tax=tax, base_path=dirname, store_f="k0k1"
    )

    np.testing.assert_equal(st.efield_arr.coords["space"].size, xax.size)
    np.testing.assert_equal(st.efield_arr.coords["time"].size, tax.size)

    np.testing.assert_equal(st.f_arr.coords["fourier_mode"].size, 2)
    np.testing.assert_equal(st.f_arr.coords["time"].size, tax.size)
    np.testing.assert_equal(st.f_arr.coords["velocity"].size, vax.size)

    shutil.rmtree(dirname)
