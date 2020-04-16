[![CircleCI](https://circleci.com/gh/joglekara/VlaPy.svg?style=shield)](https://circleci.com/gh/joglekara/VlaPy)
[![codecov](https://codecov.io/gh/joglekara/VlaPy/branch/master/graph/badge.svg)](https://codecov.io/gh/joglekara/VlaPy)
[![Documentation Status](https://readthedocs.org/projects/vlapy/badge/?version=latest)](https://vlapy.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# VlaPy

Usage details and the latest documentation can be found at https://vlapy.readthedocs.io/en/latest/

## Quick Usage
To install dependencies, run ``python3 setup.py install`` from the base directory of the repository.

After this step, ``python3 run_vlapy.py`` can be executed to run a simulation of Landau damping with collisions.

This will create a temporary directory for the simulation files. Once completed, MLFlow will move the simulation folder into a centralized datastore. This datastore can be accessed through a web-browser based UI provided by leveraging MLFlow.

To start the MLFlow UI server, type ``mlflow ui`` into the terminal and then navigate to localhost:5000 in your web browser. The page will look like the following

## Overview
VlaPy is a 1-spatial-dimension, 1-velocity-dimension, Vlasov-Poisson-Fokker-Planck code written in Python. 
The Vlasov-Poisson-Fokker-Planck system of equations is commonly used in plasma physics.

## Implementation
The Vlasov-Poisson-Fokker-Planck system can be decomposed into 4 components.

### Vlasov - Spatial Advection
The spatial advection operator is pushed pseudospectrally. The system is periodic in x. 

This operator is tested in the fully integrated tests to reproduce solutions of the 
1D-1V Vlasov-Poisson system, namely, Landau damping.

### Vlasov - Velocity Advection
The velocity advection operator is pushed pseudospectrally. The system is periodic in v.

This operator is tested in the fully integrated tests to reproduce solutions of the 
1D-1V Vlasov-Poisson system, namely, Landau damping.

 
### Poisson Solver
The Poisson equation is solved pseudospectrally. 

This solver is tested to reproduce analytical solutions to a periodic Poisson system.


### Fokker-Planck Solver
The Fokker-Planck equation is solved using an implicit finite-difference scheme because of the need to perform a 
diffusion time-step. 

This solver is tested to 
1) return df/dt = 0 if a Maxwell-Boltzmann distribution is provided as input 
2) conserve energy and density
3) relax to a Maxwellian of the right temperature and without a drift velocity

## Tests
All tests are performed in CircleCI. There are unit tests as well as integrated tests.
One of the most fundamental plasma physics phenomenon is that described by Landau damping. 

Plasmas can support electrostatic oscillations. The oscillation frequency is given by the electrostatic electron 
plasma wave (EPW) dispersion relation. When a wave of sufficiently small amplitude is driven at the resonant 
wave-number and frequency pairing, there is a resonant exchange of energy between the plasma and the electric field, 
and the electrons can damp the electric field.

In VlaPy, we verify that the damping rate is reproduced for a few different wave numbers. 
This is shown in `notebooks/landau_damping.ipynb.`

We include validation against this phenomenon as an integrated test.

## Other practical considerations
### File Storage
XArray enables a user-friendly interface to labeling multi-dimensional arrays along with a powerful and performant
backend. Therefore, we use XArray (http://xarray.pydata.org/en/stable/) for a performant Pythonic storage mechanism 
that promises lazy loading and incremental writes (through some tricks).

### Simulation Management
We use MLFlow (https://mlflow.org/) for simulation management. This is typically used for managing machine-learning
lifecycles but is perfectly suited for managing numerical simulations. We believe UI capability to manage simulations
significantly eases the physicist's workflow. 

There are more details about how the diagnostics for a particular type of simulation are packaged and provided to
the run manager object. These will be described in time. One can infer these from the code as well. 
