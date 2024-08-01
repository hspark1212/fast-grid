from setuptools import setup, find_packages

setup(
    name="fast-grid",
    version="0.2.0",
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
        "MDAnalysis",
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
)
