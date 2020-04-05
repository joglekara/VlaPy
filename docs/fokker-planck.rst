Fokker-Planck Equation
-------------------------

The Fokker-Planck equation is solved using an implicit finite-difference scheme because of the need to perform a
diffusion time-step.

This solver is tested to
1) return df/dt = 0 if a Maxwell-Boltzmann distribution is provided as input
2) conserve energy and density
3) relax to a Maxwellian of the right temperature and without a drift velocity


...this page is in development...