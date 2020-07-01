Solving the Fokker-Planck Equation
----------------------------------------

There are many approximations to the Fokker-Planck equation as applied to a distribution function of electrons.

Lenard-Bernstein Form
****************************
VlaPy contains the Lenard-Bernstein :cite:`Lenard1958` implementation of the Fokker-Planck equation. This equation is given by

.. math::
    \frac{\partial f}{\partial t} = \nu_{ee} \frac{\partial}{\partial v} \left(v f + v_0^2 \frac{\partial f}{\partial v} \right)

where :math:`\nu_{ee}` and :math:`v_0` are the electron-electron collision frequency, and thermal velocity, respectively.


Differencing Scheme
====================

This operator is differenced backwards in time, and center differenced in velocity space, which gives

.. math::
    \frac{f^{n+1}_{\alpha} - f^{n}_{\alpha}}{\Delta t} = \nu_{ee} \left[f^{n+1}_\alpha + v_\alpha \Delta_v(f^{n+1}_{\alpha}) + v_0^2 \Delta^2_v(f^{n+1}_{\alpha})\right]

where

.. math::
    \Delta_v(f^{n+1}_{\alpha})= \frac{f^{n+1}_{\alpha+1} - f^{n+1}_{\alpha-1}}{2\Delta v}

and

.. math::
    \Delta^2_v(f^{n+1}_{\alpha})= \frac{-f^{n+1}_{\alpha+1} + 2f^{n+1}_{\alpha} - f^{n+1}_{\alpha-1}}{\Delta v^2} \\


This system can be transformed into a linear system of equations in the form of

.. math::
    C_{ee} f^{n+1} = f^{n}

where :math:`C_{ee}` is a matrix that corresponds to the finite difference operator stencil. In 1D, :math:`C_{ee}`
tridiagonal matrix.  that can be solved directly.


Tests
======

This solver is tested to
1) return df/dt = 0 if a Maxwell-Boltzmann distribution is provided as input
2) conserve energy and density
3) relax to a Maxwellian of the right temperature and without a drift velocity



.. bibliography:: bibs/fokkerplanck.bib