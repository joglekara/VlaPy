Other Higher Level Tools
-------------------------


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

