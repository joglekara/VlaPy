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

    manager.start_run(
        nx=48,
        nv=512,
        nt=1000,
        tmax=100,
        nu=0.001,
        w0=1.1598,
        k0=0.3,
        a0=1e-5,
        diagnostics=landau_damping.LandauDamping(),
        name="Landau Damping",
    )

Through this interface, the user can specify the boundaries of the domain in space and time. The user can
also specify initial conditions such as the settings of the wave-driver module, the collision frequency `nu`, and custom
diagnostic routines.


Wave Driver
===============
The wave driver module currently has a fixed form given by

.. code-block:: python

    def driver_function(x, t):
        envelope = np.exp(-((t - 8) ** 8.0) / 4.0 ** 8.0)
        return envelope * a0 * np.cos(k0 * x + w0 * t)

This effectively mimics a flat-top pulse in time. The ``w0, a0,`` and ``k0`` quantities can be specified to generate waves
of specified frequency, amplitude, and wave-number, respectively.


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



...this page is in development...