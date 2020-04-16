import numpy as np

from vlapy import manager
from diagnostics import landau_damping, z_function

if __name__ == "__main__":

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
        }
    }

    pulse_dictionary["first pulse"]["w0"] = np.real(
        z_function.get_roots_to_electrostatic_dispersion(
            wp_e=1.0, vth_e=1.0, k0=pulse_dictionary["first pulse"]["k0"]
        )
    )

    manager.start_run(
        all_params=all_params_dict,
        pulse_dictionary=pulse_dictionary,
        diagnostics=landau_damping.LandauDamping(),
        name="Landau Damping-test",
    )
