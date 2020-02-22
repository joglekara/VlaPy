import os
import uuid
import shutil

from vlapy import manager
from diagnostics import landau_damping


def test_manager_folder():
    test_folder = os.path.join(os.getcwd(), str(uuid.uuid4())[-6:])
    manager.start_run(
        nx=48,
        nv=512,
        nt=100,
        tmax=10,
        nu=0.0,
        w0=1.1598,
        k0=0.3,
        a0=1e-5,
        diagnostics=landau_damping.LandauDamping(),
        name="Landau Damping",
        mlflow_path=test_folder,
    )

    assert os.path.exists(test_folder)
    shutil.rmtree(test_folder)
