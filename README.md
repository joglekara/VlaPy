[![CircleCI](https://circleci.com/gh/joglekara/VlaPy.svg?style=svg&circle-token=52a0c9c0d445766b1a051a0bb7198d2ccf0617f6)](https://circleci.com/gh/joglekara/VlaPy)
[![codecov](https://codecov.io/gh/joglekara/VlaPy/branch/master/graph/badge.svg?token=xjy3FvcZPJ)](https://codecov.io/gh/joglekara/VlaPy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# VlaPy

## Overview
VlaPy is a 1-spatial-dimension, 1-velocity-dimension, Vlasov-Poisson-Fokker-Planck code written in Python. 
The Vlasov-Poisson-Fokker-Planck system of equations is commonly used in plasma physics.

## Implementation
The Vlasov-Poisson-Fokker-Planck system can be decomposed into 4 components.

### Vlasov - Spatial Advection
The spatial advection operator is pushed pseudospectrally. The system is periodic in x. 

This operator is tested in the fully integrated tests to reproduce solutions of the 
1D-1V Vlasov-Poisson system, namely, Landau damping.

This method is accelerated using ``numba.njit``.

### Vlasov - Velocity Advection
The velocity advection operator is pushed pseudospectrally. The system is periodic in v.

This operator is tested in the fully integrated tests to reproduce solutions of the 
1D-1V Vlasov-Poisson system, namely, Landau damping.

This method is accelerated using ``numba.njit``.

 
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


