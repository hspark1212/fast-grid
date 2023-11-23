from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np


setup(
    name="fast-grid",
    version="0.1.10",
    description="Fast grid calculation",
    author="Hyunsoo Park",
    author_email="hpark@ic.ac.uk",
    url="https://github.com/hspark1212/fast-grid.git",
    install_requires=[
        "ase",
        "numba",
        "fire",
        "pandas",
        "MDAnalysis",
    ],
    entry_points={"console_scripts": ["fast-grid=fast_grid.calculate_grid:cli"]},
    packages=find_packages(),
    package_data={"fast_grid": ["assets/*.csv"]},
    include_package_data=True,
    python_requires=">=3.9",
    # cython
    setup_requires=["cython"],
    ext_modules=cythonize(
        [
            "fast_grid/potential/lj_potential.pyx",
            "fast_grid/potential/gaussian.pyx",
        ]
    ),
    include_dirs=[np.get_include()],
)
