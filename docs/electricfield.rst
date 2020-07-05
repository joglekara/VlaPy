Solving for the Electric Field
--------------------------------

To calculate the electric field at each time-step due to the free charges, we solve Poisson's Equation given by

.. math::
    - \nabla^2 \Phi(x) = \rho(x), \\
    \nabla E(x) = \rho_i(x) - \rho_e(x)

where :math:`\rho_i, \rho_e` are the charge densities for ions and electrons, respectively, and :math:`\Phi` is the
electrostatic potential. Assuming that the ion density stays uniform and constant, this can be solved in the
pseudo-spectral domain by computing the Fourier transform of the above equation. The solution in Fourier space is
given by

.. math::
    E(k_x) = \frac{1 - \rho_e(k_x)}{-i k_x}.

.. math::
    E(x) = \sum_{x={x_{min}}}^{x_{max}}\frac{1 - \rho_e(k_x)}{-i k_x} \exp{(i k_x x)}.

This solver is tested to reproduce analytical solutions to a periodic Poisson system.


