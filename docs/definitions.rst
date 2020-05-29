Glossary
------------

Discretizating continuous information
****************************************
All quantities defined over a space, velocity, and time are understood to be described
by their discretized equivalents (to the order of the truncation).

.. math::
    x = x_i,
    v = v_\alpha,
    t = t_n,

where :math:`i, \alpha, n` represent an integer index of arrays corresponding to space, velocity, and time,
respectively.

Fourier Transforming in Space and Velocity Space
*****************************************************
VlaPy relies on representing phase space in it's Fourier domain equivalent.

Given that :math:`f=f^n(x,v)` is the discretized distribution function, VlaPy uses the following definitions throughout
this documentation

.. math::
    \mathcal{F}_x(k_x, v) = \sum_{k_x=0}^{k_x=N_x/2} \exp{(-i k_x x) f(x,v)} \\
    \mathcal{F}_v(x, k_v) = \sum_{k_v=0}^{k_v=N_v/2} \exp{(-i k_v v) f(x,v)}


where :math:`\mathcal{F}_x, \mathcal{F}_v` are the discrete-Fourier-transform equivalents in configuration-space,
and velocity-space, respectively. These may also be performed simultaneously such that

.. math::
    \mathcal{F}_{x,v}(k_x, k_v) = \sum_{k_v=0}^{N_v/2} \exp{(-i k_v v)} \sum_{k_x=0}^{N_x/2} \exp{(-i k_x x)}  f(x,v).