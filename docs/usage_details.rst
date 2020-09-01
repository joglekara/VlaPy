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

    k0 = np.random.uniform(0.3, 0.4, 1)[0]
    log_nu_over_nu_ld = None

    all_params_dict = initializers.make_default_params_dictionary()
    all_params_dict = initializers.specify_epw_params_to_dict(
        k0=k0, all_params_dict=all_params_dict
    )
    all_params_dict = initializers.specify_collisions_to_dict(
        log_nu_over_nu_ld=log_nu_over_nu_ld, all_params_dict=all_params_dict
    )

    all_params_dict["vlasov-poisson"]["time"] = "leapfrog"
    all_params_dict["vlasov-poisson"]["edfdv"] = "exponential"
    all_params_dict["vlasov-poisson"]["vdfdx"] = "exponential"

    all_params_dict["fokker-planck"]["type"] = "lb"

    pulse_dictionary = {
        "first pulse": {
            "start_time": 0,
            "t_L": 6,
            "t_wL": 2.5,
            "t_R": 20,
            "t_wR": 2.5,
            "w0": all_params_dict["w_epw"],
            "a0": 1e-7,
            "k0": k0,
        }
    }

    mlflow_exp_name = "landau-damping"

    uris = {
        "tracking": "local",
    }

    print_to_screen.print_startup_message(
        mlflow_exp_name, all_params_dict, pulse_dictionary
    )

    that_run = manager.start_run(
        all_params=all_params_dict,
        pulse_dictionary=pulse_dictionary,
        diagnostics=landau_damping.LandauDamping(
            vph=all_params_dict["v_ph"],
            wepw=all_params_dict["w_epw"],
        ),
        uris=uris,
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

