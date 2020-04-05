Vlasov Equation
---------------

The 1-spatial-dimension, 1-velocity-dimension (1D-1V) Vlasov equation describing the conservation of phase space is given by

.. math::
    \frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x} + E \frac{\partial f}{\partial v} = 0

The evolution of the phase space in time can be
calculated by evolving the phase space in configuration space (i.e. in :math:`\hat{x}`) as well as evolving the phase space
in velocity space (i.e. :math:`\hat{v}_x`). For information on the definitions, please refer to the Glossary.

We split the equation into two separate steps ala Strang, or Cheng-Knorr, Splitting. This enables treating each
component as an Ordinary Differential Equation (ODE).

Time Integration
******************
Because the Vlasov equation models chaotic nonlinear dynamics, it is important to conserve the Hamiltonian in order to
be able to retrieve accurate solutions over long time-scales. VlaPy contains 2 implementations of symplectic integrators
for the Spatial and Velocity Advection operators.

The first method is a simple `Verlet`, or Leapfrog, scheme given by

.. math::
    v^{n+1/2} = v^n + a^n \frac{\Delta t}{2}, \\
    x^{n} = x^n + v^{n+1/2} \Delta t, \\
    v^{n} = v^{n+1/2} + a^{n+1} \frac{\Delta t}{2}.

This is a second order method i.e. the truncation error scales as :math:`\mathcal{O}(\Delta t^2)`

The fourth order implementation is based on the `Position-Extended Forest-Ruth-Like` (PEFRL) algorithm. Details on this
implementation will be added soon.

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


Velocity Advection
*******************
The velocity advection operator is

.. math::
    \frac{\partial f}{\partial t} = E \frac{\partial f}{\partial v}

In VlaPy, the domain is assumed to be periodic in velocity. While this constraint is not fulfilled at the boundaries,
i.e. the first derivative is not continuous across the boundary, the value of the distribution function, and it's
derivatives can be very small if  :math:`\text{(|v_{min}|, v_{max}) > 6 v_{th}`.

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


...this page is in development...