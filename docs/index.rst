.. VlaPy documentation master file, created by
   sphinx-quickstart on Sun Mar 29 08:24:59 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to VlaPy!
=================================

Overview
---------
VlaPy is a 1-spatial-dimension, 1-velocity-dimension, Vlasov-Poisson-Fokker-Planck code written in Python.
The Vlasov-Poisson-Fokker-Planck system of equations is commonly used in plasma physics.

The implementation details are given in the following pages

.. toctree::
   :maxdepth: 2

   vlasov
   fokker-planck
   electricfield
   definitions




Other practical considerations
---------------------------------
File Storage
************
XArray enables a user-friendly interface to labeling multi-dimensional arrays along with a powerful and performant
backend. Therefore, we use XArray (http://xarray.pydata.org/en/stable/) for a performant Python storage library that
leverages NetCDF and promises lazy loading and incremental writes.

Simulation Management
*********************
We use MLFlow (https://mlflow.org/) for simulation management. This is typically used for managing machine-learning
lifecycles but is perfectly suited for managing numerical simulations. We believe UI capability to manage simulations
significantly eases the physicist's workflow.

There are more details about how the diagnostics for a particular type of simulation are packaged and provided to
the run manager object. These will be described in time. One can infer these details from the code as well.


Contribution
---------------
If you would like to contribute, please raise an issue in the GitHub repository using the issue tracker. The repository
is located at https://github.com/joglekara/vlapy.


...this page is in development...




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
