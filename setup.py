from setuptools import setup, find_packages

INSTALL_REQUIREMENTS = [
    "numpy",
    "numba",
    "mlflow",
    "xarray",
    "scipy",
    "matplotlib",
]

setup(
    name="vlapy",
    version="1.0",
    packages=find_packages(),
    url="",
    license="MIT",
    author="A. S. Joglekar, M. C. Levy",
    author_email="archisj@gmail.com",
    description="Pseudo-Spectral, Modular, Pythonic 1D-1V Vlasov-Fokker-Planck code",
    install_requires=INSTALL_REQUIREMENTS,
    include_package_data=True,
)
