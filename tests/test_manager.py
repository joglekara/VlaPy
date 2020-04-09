import os
import uuid
import shutil
import numpy as np

from vlapy import manager
from diagnostics import landau_damping


def test_manager_folder():
    test_folder = os.path.join(os.getcwd(), str(uuid.uuid4())[-6:])
    all_params_dict = {
        "nx": 48,
        "xmin": 0.0,
        "xmax": 2.0 * np.pi / 0.3,
        "nv": 512,
        "vmax": 6.0,
        "nt": 1000,
        "tmax": 100,
        "nu": 0.0,
    }

    pulse_dictionary = {
        "first pulse": {
            "start_time": 0,
            "rise_time": 5,
            "flat_time": 10,
            "fall_time": 5,
            "a0": 1e-6,
            "k0": 0.3,
            "w0": 1.1598,
        }
    }

    manager.start_run(
        all_params=all_params_dict,
        pulse_dictionary=pulse_dictionary,
        diagnostics=landau_damping.LandauDamping(),
        name="Landau Damping",
        mlflow_path=test_folder,
    )

    assert os.path.exists(test_folder)
    shutil.rmtree(test_folder)
