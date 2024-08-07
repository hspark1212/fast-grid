
<div align="center">

<h1> Fast Grid 🏁 </h1>

  <p>
    <strong>High-speed Voxel Grid Calculations</strong>
  </p>

</div>

## Features

### Supported potentials
- LJ Potential (Lennard-Jones)
- Gaussian Potential

### Gas Probe Model
- TraPPE (Methane) [Default]

## Installation

To install Fast Grid, run the following command in your terminal:

```bash
pip install fast-grid
```

## Quick Start
- Help Command

```bash
fast-grid --help
```

- Run an example to generaate enegy grid with the LJ potential

```bash
fast-grid --atoms=examples/irmof-1.cif --visualize=True
```

Check out a [tutorial](tutorial.ipynb) file for more details

## Usage

### 1. LJ potential

Calculate a 30x30x30 energy grid using the LJ potential:

```python
from fast_grid import calculate_grid

calculate_grid(
    atoms="examples/irmof-1.cif",
    grid_size=30,
    ff_type="UFF",
    potential="LJ",
    cutoff=12.8,
    gas_epsilon=148.0,
    gas_sigma=3.73,
    visualize=True,
)
```

- UFF Force Field for atoms in the structure
- Cutoff: 12.8 Å
- Gas Probe Parameters: TraPPE for methane united atom model

![lj_irmof-1](./images/lj_example.png)
 
### 2. Gaussian potential

Calculate a voxel grid with the Gaussian function:

```python
from fast_grid import calculate_grid
from ase.build import bulk

atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
calculate_grid(
    atoms=atoms,
    grid_spacing=0.2,
    potential="Gaussian",
    gaussian_height=1.0,
    gaussian_width=1.0,
    visualize=True,
    pallete="atomic",
)
```

- Gaussian Parameters: Height - 1.0, Width - 1.0

![gaussian_irmof-1](./images/gaussian_example.png)
