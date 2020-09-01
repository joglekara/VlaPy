Solving the Vlasov Equation
------------------------------

The 1-spatial-dimension, 1-velocity-dimension (1D-1V) Vlasov equation describing the conservation of phase space is given by

.. math::
    \frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x} + E \frac{\partial f}{\partial v} = 0

The evolution of the phase space in time can be
calculated by evolving the phase space in configuration space (i.e. in :math:`\hat{x}`) as well as evolving the phase space
in velocity space (i.e. :math:`\hat{v}_x`). For information on the definitions, please refer to the Glossary.

We split the equation into two separate steps ala Strang :cite:`Strang1968`, or Cheng-Knorr :cite:`Cheng1976`, Splitting. This enables treating each
component as an Ordinary Differential Equation (ODE).

Time Integration
******************
Because the Vlasov equation models chaotic nonlinear dynamics, it is important to conserve the Hamiltonian in order to
be able to retrieve accurate solutions over long time-scales. VlaPy contains 3 implementations of symplectic integrators
for the Spatial and Velocity Advection operators.

The first method is a simple `Verlet`, or Leapfrog, scheme given by

.. math::
    v^{n+1/2} = v^n + a^n \frac{\Delta t}{2}, \\
    x^{n} = x^n + v^{n+1/2} \Delta t, \\
    v^{n} = v^{n+1/2} + a^{n+1} \frac{\Delta t}{2}.

This is a second order method i.e. the truncation error scales as :math:`\mathcal{O}(\Delta t^2)`

The fourth order implementation is based on the `Position-Extended Forest-Ruth-Like` (PEFRL) :cite:`Omelyan2002`
algorithm.

The sixth order implementation is based on work in :cite:`Casas2017`

Spatial Advection
******************
The spatial advection operator is

.. math::
    \frac{\partial f}{\partial t} = v \frac{\partial f}{\partial x}

In VlaPy, the domain is periodic in space. This constraint enables the use of Fourier decomposition methods such that
the operator can be integrated pseudo-spectrally in space. Computing a Fourier Transform of the above equation gives
the following ODE

.. math::
    \frac{d F_x}{d t} = v \left[(-i k_x) F_x\right], \\
    \frac{d F_x}{F_x} = v (-i k_x) dt.

The solution is obtained by discretizing the above in time and is given by

.. math::
    F_x^{n+1} = F_x^n \exp{[v (-i k_x) \Delta t]}.

Computing the inverse Fourier transform of the above, and only keeping the real part, gives the evolved distribution
function

.. math::
    f^{n+1} = \text{Real}\left[\sum_{x_i = x_{min}}^{x_{max}} \exp{(i k_x x)} F_x^{n+1}\right]

This method is inspired by conversation and notes from Pierre Navaro (https://github.com/pnavaro). It is chosen for
it's simplicity as well as accuracy.

We have also implemented and tested the Backward Semi-Lagrangian Operator :cite:`Cheng1976`.

Velocity Advection
*******************
The velocity advection operator is

.. math::
    \frac{\partial f}{\partial t} = E \frac{\partial f}{\partial v}

In VlaPy, the domain is assumed to be periodic in velocity. While this constraint is not fulfilled at the boundaries,
i.e. the first derivative is not continuous across the boundary, the value of the distribution function, and it's
derivatives can be very small if  :math:`(|v_{min}|, v_{max}) > 6 v_{th}`.

This constraint enables the use of Fourier decomposition methods such that the operator can be integrated
pseudo-spectrally in velocity-space. Computing a Fourier Transform of the above equation gives the following ODE

.. math::
    \frac{d F_v}{d t} = E \left[(-i k_v) F_v\right], \\
    \frac{d F_v}{F_v} = E (-i k_v) dt.

The solution is obtained by discretizing the above in time and is given by

.. math::
    F_v^{n+1} = F_v^{n} \exp{[E (-i k_v) \Delta t]}.

Computing the inverse Fourier transform of the above, and only keeping the real part, gives the evolved distribution
function

.. math::
    f^{n+1} = \text{Real}\left[\sum_{v_\alpha = v_{min}}^{v_{max}} \exp{(i k_v v)} F_v^{n+1}\right]



This method is inspired by conversation and notes from Pierre Navaro (https://github.com/pnavaro). It is chosen for
it's simplicity as well as accuracy.

We have also implemented and tested the Backward Semi-Lagrangian Operator :cite:`Cheng1976` and a 2nd-order
centered-difference Operator.

Tests
******

One of the most fundamental plasma physics phenomenon is that described by Landau damping :cite:`Ryutov1999`.

Plasmas can support electrostatic oscillations. The oscillation frequency is given by the electrostatic electron
plasma wave (EPW) dispersion relation. When a wave of sufficiently small amplitude is driven at the resonant
wave-number and frequency pairing, there is a resonant exchange of energy between the plasma and the electric field,
and the electrons can damp the electric field.

In VlaPy, we verify that the damping rate is reproduced for a few different wave numbers.
This is shown in `notebooks/landau_damping.ipynb.`

.. bibliography:: bibs/vlasov.bib
    :style: unsrtalpha