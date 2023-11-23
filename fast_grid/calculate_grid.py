import warnings
from typing import Union, Iterable
from pathlib import Path

from fire import Fire

import numpy as np
from ase import Atoms
from ase.io import read

from MDAnalysis.lib.distances import distance_array as mic_distance_matrix

from fast_grid.ff import get_mixing_epsilon_sigma
<<<<<<< HEAD:fast_grid/calculate_grid.py
from fast_grid.potential import calculate_lj_potential, calculate_gaussian
=======
from fast_grid.utils import check_inputs_energy_grid
from fast_grid.potential import lj_potential_cython, gaussian_cython
>>>>>>> 3078b916b4982a35e30c91e6df77c76480bf4184:fast_grid/calculate_grid.py
from fast_grid.visualize import visualize_grid

warnings.filterwarnings("ignore")


def calculate_grid(
    structure: Union[Atoms, str],
    grid_size: Union[int, Iterable] = 30,
    grid_spacing: float = None,
    ff_type: str = "UFF",
    potential: str = "LJ",
    cutoff: float = 12.8,
    gas_epsilon: float = 148.0,  # LJ
    gas_sigma: float = 3.73,  # LJ
    visualize: bool = False,
<<<<<<< HEAD:fast_grid/calculate_grid.py
    gaussian_height: float = 0.1,
    gaussian_width: float = 5.0,
=======
    gaussian_height: float = 0.1,  # Gaussian
    gaussian_width: float = 5.0,  # Gaussian
>>>>>>> 3078b916b4982a35e30c91e6df77c76480bf4184:fast_grid/calculate_grid.py
    float16: bool = False,
    emax: float = 5000.0,
    emin: float = -5000.0,
    output_shape_grid: bool = False,
) -> np.array:
    """Calculate the energy grid for a given structure and force field.
    It takes a structure (ase Atoms object or cif file path) and returns the energy grid.
    The supported potentials are Lennard-Jones and Gaussian.
    The supported force field is UFF.
    The gas parameters are for methane in TraPPE-UA (united-atoms) force field.
    The unit of energy is K and the unit of distance is Angstrom.
    The output shape of energy grid is grid_size * grid_size * grid_size.

    :param structure: structure (ase Atoms object or cif file path)
    :param grid_size: grid size, for example, 30 or "(30, 30, 30)", defaults to 30
    :param grid_spacing: grid spacing, overrides grid_size, defaults to None
    :param ff_type: force field type, defaults to "UFF"
    :param potential: potential function, gaussian or lj, defaults to "LJ"
    :param cutoff: cutoff distance, defaults to 12.8
    :param gas_epsilon: gas epsilon, in K (methane UA in TraPPE), defaults to 148.0
    :param gas_sigma: gas sigma, in Angstrom (methane UA in TraPPE), defaults to 3.73
    :param visualize: visualize the energy grid, defaults to False
    :param gaussian_height: gaussian height
    :param gaussian_width: gaussian width
    :param float16: use float16 to save memory, defaults to False
    :param emax: clip energy values for better visualization, defaults to 5000.0
    :param emin: clip energy values for better visualization, defaults to -5000.0
    :param output_shape_grid: output shape of energy grid, defaults to False
    :return: energy grid
    """
    # read structure
    if isinstance(structure, Atoms):
        atoms = structure
    elif isinstance(structure, str):
        if Path(structure).exists():
            atoms = read(structure)
        else:
            raise FileNotFoundError(f"{structure} does not exist")
    else:
        raise TypeError("structure must be an ase Atoms object or a cif file path")

    # make supercell when distance between planes is less than cutoff * 2
    cell_volume = atoms.get_volume()
    cell_vectors = np.array(atoms.cell)
    dist_a = cell_volume / np.linalg.norm(np.cross(cell_vectors[1], cell_vectors[2]))
    dist_b = cell_volume / np.linalg.norm(np.cross(cell_vectors[2], cell_vectors[0]))
    dist_c = cell_volume / np.linalg.norm(np.cross(cell_vectors[0], cell_vectors[1]))
    plane_distances = np.array([dist_a, dist_b, dist_c])
    supercell = np.ceil(2 * cutoff / plane_distances).astype(int)
<<<<<<< HEAD:fast_grid/calculate_grid.py
    atoms = atoms.repeat(supercell)  # make supercell

=======
    atoms = atoms.repeat(supercell)
>>>>>>> 3078b916b4982a35e30c91e6df77c76480bf4184:fast_grid/calculate_grid.py
    cell_vectors = np.array(atoms.cell)  # redefine cell_vectors after supercell

    # get position for grid
    if isinstance(grid_size, int):
        grid_size = np.array([grid_size] * 3)
    else:
        grid_size = eval(grid_size)

    # override grid_size if grid_spacing is not None
    if grid_spacing is not None:
        grid_size = np.ceil(
            np.array(atoms.get_cell_lengths_and_angles()[:3]) / grid_spacing
        ).astype(int)
    assert len(grid_size) == 3, "grid_size must be a 3-dim vector"

<<<<<<< HEAD:fast_grid/calculate_grid.py
    indices = np.indices(grid_size).reshape(3, -1).T
=======
    indices = np.indices(grid_size).reshape(3, -1).T  # (G, 3)
>>>>>>> 3078b916b4982a35e30c91e6df77c76480bf4184:fast_grid/calculate_grid.py
    pos_grid = indices.dot(cell_vectors / grid_size)  # (G, 3)

    # get positions for atoms
    pos_atoms = atoms.get_positions()  # (N, 3)

    # setting force field
    symbols = atoms.get_chemical_symbols()
    epsilon, sigma = get_mixing_epsilon_sigma(
        symbols, ff_type, gas_epsilon, gas_sigma
    )  # (N,) (N,)

<<<<<<< HEAD:fast_grid/calculate_grid.py
    # calculate distance matrix
    box = atoms.cell.cellpar()
    dist_matrix = mic_distance_matrix(pos_grid, pos_atoms, box)  # (G, N)

    # calculate energy
    if potential.lower() == "lj":
        calculated_grid = calculate_lj_potential(
            dist_matrix,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
        )  # (G,)
    elif potential.lower() == "gaussian":
        calculated_grid = calculate_gaussian(
            dist_matrix,
            height=gaussian_height,
            width=gaussian_width,
            cutoff=cutoff,
        )  # (G,)
=======
    # check inputs for energy grid
    inverse_cell = np.linalg.inv(cell_vectors)
    energy_grid = np.zeros([grid_size[0] * grid_size[1] * grid_size[2]])
    check_inputs_energy_grid(
        pos1=pos_grid,
        pos2=pos_atoms,
        cell_vectors=cell_vectors,
        inverse_cell=inverse_cell,
        cutoff=cutoff,
        energy_grid=energy_grid,
        epsilon=epsilon,
        sigma=sigma,
        gaussian_height=gaussian_height,
        gaussian_width=gaussian_width,
    )

    # calculate energy
    if potential.lower() == "lj":
        calculated_grid = lj_potential_cython(
            pos_grid,
            pos_atoms,
            cell_vectors,
            inverse_cell,
            epsilon,
            sigma,
            cutoff,
            energy_grid,
        )  # (G, 3)
    elif potential.lower() == "gaussian":
        calculated_grid = gaussian_cython(
            pos_grid,
            pos_atoms,
            cell_vectors,
            inverse_cell,
            gaussian_height,
            gaussian_width,
            cutoff,
            energy_grid,
        )  # (G, 3)
>>>>>>> 3078b916b4982a35e30c91e6df77c76480bf4184:fast_grid/calculate_grid.py
    else:
        raise NotImplementedError(f"{potential} should be one of ['LJ', 'Gaussian']")

    # flatten energy grid
    calculated_grid = calculated_grid.reshape(-1)  # (G,)

    # convert to float16 to save memory
    if float16:
        # clip energy values for np.float16
        min_float16 = np.finfo(np.float16).min
        max_float16 = np.finfo(np.float16).max
        calculated_grid = np.clip(calculated_grid, min_float16, max_float16)
        # convert to float16
        calculated_grid = calculated_grid.astype(np.float16)
<<<<<<< HEAD:fast_grid/calculate_grid.py

    if visualize:
        print(f"Visualizing energy grid | supercell {supercell}...")
=======

    if output_shape_grid:
        return calculated_grid.reshape(grid_size)

    if visualize:
        print(
            f"Visualizing energy grid with {grid_size} grid points | supercell: {supercell}"
        )
>>>>>>> 3078b916b4982a35e30c91e6df77c76480bf4184:fast_grid/calculate_grid.py
        visualize_grid(pos_grid, pos_atoms, calculated_grid, emax, emin)

    return calculated_grid


def cli():
    Fire(calculate_grid)
