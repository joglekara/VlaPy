# MIT License
#
# Copyright (c) 2020 Archis Joglekar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from setuptools import setup, find_packages

INSTALL_REQUIREMENTS = [
    "wheel",
    "numpy",
    "mlflow",
    "h5netcdf",
    "xarray",
    "scipy",
    "matplotlib",
    "tqdm",
    "dask[complete]",
]

setup(
    name="vlapy",
    version="0.1",
    packages=find_packages(),
    url="",
    license="MIT",
    author="A. S. Joglekar, M. C. Levy",
    author_email="archisj@gmail.com",
    description="Pseudo-Spectral, Modular, Pythonic 1D-1V Vlasov-Fokker-Planck code",
    install_requires=INSTALL_REQUIREMENTS,
    include_package_data=True,
)
