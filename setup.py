from setuptools import setup, find_packages

setup(
    name="fast-grid",
    version="0.1.4",
    description="Fast grid calculation",
    author="Hyunsoo Park",
    author_email="hpark@ic.ac.uk",
    url="https://github.com/hspark1212/fast-grid.git",
    install_requires=[
        "numpy",
        "ase",
        "numba",
        "fire",
        "plotly",
        "pandas",
    ],
    entry_points={"console_scripts": ["fast-grid=fast_grid.calculate_grids:cli"]},
    packages=find_packages(),
    package_data={"fast_grid": ["assets/*.csv"]},
    include_package_data=True,
    python_requires=">=3.9",
)
