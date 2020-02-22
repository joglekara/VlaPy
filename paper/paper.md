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

There is a plethora of software that solves the same equation set in academia [@Banks2017, @Joglekar2018], research labs, and industry, but a simple-to-read, open-source Python implementation is still lacking. This lack of simulation capability is echoed by the ``PlasmaPy`` [@plasmapy] community (``PlasmaPy`` is a collection of Open-Source plasma physics resources). ``VlaPy`` aims to fulfill these voids in the academic and research communities.

In general, ``VlaPy`` is designed to help students and researchers learn about concepts such as fundamental plasma physics and numerical methods as well as software-engineering-related topics such as unit and integrated testing, and extensible and maintainable code. The details of the implementation are provided in the following section. 