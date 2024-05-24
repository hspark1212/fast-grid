from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

setup(
    name="fast-grid",
    version="0.1.15",
    description="Fast grid calculation",
    author="Hyunsoo Park",
    author_email="phs68660888@gmail.com",
    url="https://github.com/hspark1212/fast-grid.git",
    install_requires=[
        "ase",
        "numba",
        "fire",
        "pandas",
        "plotly",
        "ipykernel",
        "nbformat",
    ],
    entry_points={"console_scripts": ["fast-grid=fast_grid.calculate_grid:cli"]},
    packages=find_packages(),
    package_data={"fast_grid": ["assets/*.csv"]},
    include_package_data=True,
    python_requires=">=3.9",
    # readme
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    # cython
    setup_requires=["cython", "numpy"],
    ext_modules=cythonize(
        [
            "fast_grid/libs/distance_matrix.pyx",
            "fast_grid/libs/potential.pyx",
        ],
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=[np.get_include()],
    extras_require={
        "pypy": ["pypy-cffi"]
    },  # Additional configurations for PyPy compatibility
)
