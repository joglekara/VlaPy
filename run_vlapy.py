from vlapy import manager
from diagnostics import landau_damping

if __name__ == "__main__":

    manager.start_run(
        nx=48,
        nv=512,
        nt=1000,
        tmax=100,
        nu=0.001,
        w0=1.1598,
        k0=0.3,
        a0=1e-5,
        diagnostics=landau_damping.LandauDamping(),
        name="Landau Damping - test",
    )
