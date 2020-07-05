Detailed Usage Notes
---------------------

Installation
***************

The dependencies can be installed into your currently active python3 environment using the command
``python3 setup.py install``

Execution
************

VlaPy can be executed using the ``start_run`` method within the ``manager`` module.
For example, `run_vlapy.py` has the following code

.. code-block:: python

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

    params_to_log = ["w0", "k0", "a0"]

    pulse_dictionary["first pulse"]["w0"] = np.real(
        z_function.get_roots_to_electrostatic_dispersion(
            wp_e=1.0, vth_e=1.0, k0=pulse_dictionary["first pulse"]["k0"]
        )
    )

    mlflow_exp_name = "Landau Damping-test"

    manager.start_run(
        all_params=all_params_dict,
        pulse_dictionary=pulse_dictionary,
        diagnostics=landau_damping.LandauDamping(params_to_log),
        name=mlflow_exp_name,
    )

Through this interface, the user can specify the initial conditions of the domain in space and time. The user can
also specify the settings of the wave-driver module, the collision frequency `nu`, and custom
diagnostic routines.


Wave Driver
===============
The wave driver module uses a 5th order polynomial that is given in ref. [@Joglekar2018]. The implementation
can be found in `vlapy/manager.py`.


Simulation Management
**********************
VlaPy leverages MLflow to manage data corresponding to different categories and types of simulations.
Before starting a simulation, VlaPy enables you to provide an experiment name (in MLFlow parlance).
Each simulation run with this experiment name will be stored here.

Each simulation creates a temporary directory for the simulation files. Once completed, MLFlow will move the simulation
folder into a centralized datastore corresponding to the specified experiment name. This datastore can be accessed
through a web-browser based UI.

To start the MLFlow UI server, type ``mlflow ui`` into the terminal and then navigate to ``localhost:5000`` in your
web browser. If you have run the default ``run_vlapy.py`` script a few times, you will see a page like this one below

.. image:: images/mlflow_screenshot.png
   :width: 900


Diagnostics
************
The diagnostics module is designed to provide flexibility to the user. The user is free to design their own diagnostics
that are called at the end of the simulation. The current implementation relies on a :code:`diagnostics.<CUSTOMCLASS>(storage_manager)`
call that performs all the necessary diagnostics.

For example, please refer to the Landau damping diagnostics in `diagnostics/landau_damping.py` where the electric field
damping rate and oscillation frequency are calculated, and plots are made of the time-evolution of the electric field
to be eventually stored by the run manager object in a location of its choosing.

