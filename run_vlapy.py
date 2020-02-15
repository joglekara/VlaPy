from vlapy import manager
from diagnostics import landau_damping
import os


if __name__ == "__main__":
    temp_dir = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_dir, exist_ok=True)

    manager.start_run(
        temp_dir,
        nx=48,
        nv=512,
        nt=1000,
        tmax=100,
        nu=0.,
        w0=1.1598,
        k0=0.3,
        a0=1e-5,
        diagnostics=landau_damping.LandauDamping(),
        name="Landau Damping"
    )
