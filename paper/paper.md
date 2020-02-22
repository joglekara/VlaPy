---
title: 'VlaPy: A Python package for Eulerian Vlasov-Poisson-Fokker-Planck Simulations'
tags:
  - Python
  - plasma physics
  - dynamics
  - astrophysics
  - fusion
authors:
  - name: Archis S. Joglekar
    orcid: 0000-0003-3599-5629
    affiliation: "1"
  - name: Matthew C. Levy
    orcid: 0000-0002-7387-0256
    affiliation: "1"
affiliations:
 - name: Noble AI, San Francisco, CA
   index: 1
date: 16 February 2020
bibliography: paper.bib

---


# Summary

``VlaPy`` is a 1-spatial-dimension, 1-velocity-dimension, Vlasov-Poisson-Fokker-Planck simulation code written in Python.  The Vlasov-Poisson-Fokker-Planck system of equations is commonly used in studying plasma physics in a variety of settings ranging from space physics to laboratory-created plasmas for fusion applications. 

The Vlasov-Poisson system is used to model collisionless plasmas. The Fokker-Planck operator is used to represent the effect of collisions. Rather than relying on numerical diffusion to smooth small-scale structures that inevitably arise when modeling collisionless plasmas, the Fokker-Planck equation enables a physical smoothing mechanism. 

The implementation here is based on finite-difference and pseudo-spectral methods. At the lowest level, ``VlaPy`` evolves a 2D grid in time according a set of coupled partial integro-differential equations over time. The dynamics are initialized through initial conditions or through an external force.

# Statement of Need

There is a plethora of software that solves the same equation set in academia (see [@Banks2017],[@Joglekar2018]), research labs, and industry, but a simple-to-read, open-source Python implementation is still lacking. This lack of simulation capability is echoed by the ``PlasmaPy`` [@plasmapy] community (``PlasmaPy`` is a collection of Open-Source plasma physics resources). ``VlaPy`` aims to fulfill these voids in the academic and research communities.

In general, ``VlaPy`` is designed to help students and researchers learn about concepts such as fundamental plasma physics and numerical methods as well as software-engineering-related topics such as unit and integrated testing, and extensible and maintainable code. The details of the implementation are provided in the following section. 


# Equations

The Vlasov-Poisson-Fokker-Planck system can be decomposed into 4 components. The normalized quantities are 
$\tilde{v} = v/v_{th}$, $\tilde{t} = t / \omega_p$, $\tilde{x} = x / (v_{th} / \omega_p)$, $\tilde{m} = m / m_e$, $\tilde{E} = e E / m_e$, $\tilde{f} = f / m n_e v_{th}^3$. The Fourier Transform operator is represented by $\mathcal{F}$. The subscript to the operator indicates the dimension of the transform. 

## Vlasov Equation

The normalized Vlasov equation is given by
$$ \frac{\partial f}{\partial t} + v  \frac{\partial f}{\partial x} + E \frac{\partial f}{\partial v} = 0 $$.

We use operator splitting to advance the time-step `@Cheng:1977`. Each one of those operators is then integrated pseudo-spectrally using the following methodology.

We first Fourier transform the operator like in 
$$ \mathcal{F}_x\left[ \frac{d f}{d t} = v \frac{d f}{d x} \right].$$
Then we solve for the change in the distribution function, discretize, and integrate, as given by
$$\frac{d\hat{f}}{\hat{f}} = v~ (-i k_x)~ dt, $$
$$ \hat{f}^{n+1}(k_x, v) = \exp(-i k_x ~ v \Delta t) ~~ \hat{f}^n(k_x, v). $$ 

The $E \partial f/\partial v$ term is stepped similarly using
$$ \hat{f}^{n+1}(x, k_v) = \exp(-i k_v ~ F \Delta t) ~~ \hat{f}^n(x, k_v) $$

We have implemented a simple Leapfrog scheme as well as a 4th order integrator called the 
Position-Extended-Forest-Ruth-Like Algorithm (PEFRL) [@Omelyan2002]

### Tests
The implementation of this equation is tested in the integrated tests section below.

## Poisson Equation

The normalized Poisson equation is
$$  \nabla^2 \Phi = \rho $$

We choose to reframe the above equation as
$$ - \nabla E = \rho_{net} = 1 - \rho_e $$ 

because the ions are motionless and form a charge-neutralizing background. This is justifiable on time-scales that are 
mall relative to the dominant time-scale for ion motion.

In 1 spatial dimension, this turns into

$$ - \frac{d}{dx} E(x) = 1 - \int f(x,v) ~dv $$

and the discretized version that is solved is

$$  E(x_i)^{n+1} = \mathcal{F}_x^{-1}\left[\frac{\sum_j f(x_i,v_j)^n \Delta v}{- i k_x}\right] $$

#### Tests
This operator has unit-tests associated with it which are simply unit tests against analytical solutions of integrals of periodic functions.


## Fokker-Planck Equation

We use a simplified version of the full Fokker-Planck operator [@Lenard1958]. This is given by

$$\left(\frac{\delta f}{\delta t}\right)_{\text{coll}} = \nu \frac{\partial}{\partial v} \left ( v f + v_0^2 \frac{\partial f}{\partial v}\right), $$
where $v_0$ is the thermal velocity of the Maxwell-Boltzmann distribution that is the solution to this equation.

We discretize this backward-in-time, centered-in-space, that results in the time-step scheme given by
$$ f^{n} = {\Delta t} \nu \left[\left(-\frac{v_0^2}{\Delta v^2} + \frac{1}{2\Delta v}\right) v_{j+1}f^{n+1}_{j+1} + \left(1+2\frac{v_0^2}{\Delta v^2}\right) f^{n+1}_j + \left(-\frac{v_0^2}{\Delta v^2} - \frac{1}{2\Delta v}\right) v_{j-1}f^{n+1}_{j-1}  \right]. $$ 

This forms a tridiagonal system of equations that can be directly inverted.

#### Tests
This operator has unit-tests associated with it. The unit tests ensure that

1. The operator conserves density.


2. The operator reverts to a solution with a temperature $v_0^2$.


3. The operator does nothing when a Maxwell-Boltzmann distribution with $v_{th} = v_0$ is fed to it.


4. The operator returns the distribution to a mean velocity of 0 if initialized with a drift velocity off-center.

# Integrated test against Plasma Physics - Electron Plasma Waves and Landau Damping

One of the most fundamental plasma physics phenomenon is that described by Landau damping. An extensive review is provided in ref. [@Ryutov1999]. 

Plasmas can support electrostatic oscillations. The oscillation frequency is given by the electrostatic electron plasma wave (EPW) dispersion relation. When a wave of sufficiently small amplitude is driven at the resonant wave-number and frequency pairing, there is a resonant exchange of energy between the plasma and the electric field, and the electrons can damp the electric field. The damping rates, as well as the resonant frequencies, are given in ref. [@Canosa:1973].

In ``VlaPy``, we verify that the damping rate is reproduced for a few different wave numbers. This is shown in `notebooks/landau_damping.ipynb`. 

We include validation against this phenomenon as an integrated test.

# Acknowledgements
We acknowledge valuable discussions with Pierre Navarro on the implementation of the Vlasov equation.

