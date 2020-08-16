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
import tempfile

import numpy as np

from vlapy import storage, inner_loop
from tests import helpers


def __initialize_base_storage_stuff__(td, xax, vax, rules_to_store_f=None):
    if rules_to_store_f is None:
        rules_to_store_f = {"space": "all", "time": "all"}

    st = storage.StorageManager(
        xax=xax,
        vax=vax,
        base_path=td,
        rules_to_store_f=rules_to_store_f,
        all_params={},
        pulse_dictionary={},
        num_steps_in_one_loop=2,
    )

    stuff_for_time_loop = {
        "f": helpers.__initialize_f__(nx=xax.size, v=vax, v0=1.0, vshift=0.0),
        "e": np.ones(xax.size),
    }

    sim_config = inner_loop.get_arrays_for_time_loop(
        stuff_for_time_loop=stuff_for_time_loop,
        nt_in_loop=2,
        store_f_rules=rules_to_store_f,
    )

    st.batch_update(sim_config=sim_config)

    return st


def test_storage_individual_file_creation():
    xax = np.linspace(0, 1, 16)
    vax = np.linspace(-6, 6, 24)

    with tempfile.TemporaryDirectory() as td:
        st = __initialize_base_storage_stuff__(td, xax, vax)

        assert os.path.exists(os.path.join(st.paths["fields-individual"], "000.nc"))
        assert os.path.exists(
            os.path.join(st.paths["distribution-individual"], "000.nc")
        )


def test_storage_total_file_creation():
    xax = np.linspace(0, 1, 16)
    vax = np.linspace(-6, 6, 24)

    with tempfile.TemporaryDirectory() as td:
        st = __initialize_base_storage_stuff__(td, xax, vax)
        st.load_data_over_all_timesteps()

        assert os.path.exists(os.path.join(st.paths["fields"], "all-fields.nc"))
        assert os.path.exists(
            os.path.join(st.paths["distribution"], "all-distribution.nc")
        )


def test_storage_init_shape():
    xax = np.linspace(0, 1, 16)
    vax = np.linspace(-6, 6, 24)

    with tempfile.TemporaryDirectory() as td:
        st = __initialize_base_storage_stuff__(td, xax, vax)
        st.load_data_over_all_timesteps()
        np.testing.assert_equal(st.fields_dataset["e"].coords["space"].size, xax.size)
        np.testing.assert_equal(
            st.dist_dataset["distribution_function"].coords["space"].size, xax.size
        )
        np.testing.assert_equal(
            st.dist_dataset["distribution_function"].coords["velocity"].size, vax.size
        )


def test_storage_init_shape_fourier():
    xax = np.linspace(0, 1, 16)
    vax = np.linspace(-6, 6, 24)

    with tempfile.TemporaryDirectory() as td:
        rules_to_store_f = {"space": ["k0", "k1"], "time": "all"}
        st = __initialize_base_storage_stuff__(
            td, xax, vax, rules_to_store_f=rules_to_store_f
        )

        st.load_data_over_all_timesteps()

        np.testing.assert_equal(st.fields_dataset["e"].coords["space"].size, xax.size)
        np.testing.assert_equal(
            st.dist_dataset["distribution_function"].coords["fourier_mode"].size,
            len(st.rules_to_store_f["space"]),
        )
        np.testing.assert_equal(
            st.dist_dataset["distribution_function"].coords["velocity"].size, vax.size
        )
