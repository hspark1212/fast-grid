import warnings
from typing import Union, Iterable
from pathlib import Path

from fire import Fire

import numpy as np
from ase import Atoms
from ase.io import read

from fast_grid.ff import get_mixing_epsilon_sigma
from fast_grid.utils import mic_distance_matrix
from fast_grid.potential import calculate_lj_potential, calculate_gaussian
from fast_grid.visualize import visualize_grids

warnings.filterwarnings("ignore")


def calculate_grids(
    structure: Union[Atoms, str],
    grid_size: Union[int, Iterable] = 30,
    ff_type: str = "UFF",
    potential: str = "LJ",
    cutoff: float = 12.8,
    gas_epsilon: float = 148.0,
    gas_sigma: float = 3.73,
    visualize: bool = False,
    max_num_atoms: int = 3000,
    gaussian_height: float = 0.1,
    gaussian_width: float = 5.0,
):
    """Calculate the energy grid for a given structure and force field.
    It takes a structure (ase Atoms object or cif file path) and returns the energy grid.
    The supported potentials are Lennard-Jones and Gaussian.
    The supported force field is UFF.
    The gas parameters are for methane in TraPPE-UA (united-atoms) force field.
    The unit of energy is K and the unit of distance is Angstrom.
    The output shape of energy grid is grid_size * grid_size * grid_size.

    :param structure: structure (ase Atoms object or cif file path)
    :param grid_size: grid size, for example, 30 or "(30, 30, 30)", defaults to 30
    :param ff_type: force field type, defaults to "UFF"
    :param potential: potential function, gaussian or lj, defaults to "LJ"
    :param cutoff: cutoff distance, defaults to 12.8
    :param gas_epsilon: gas epsilon, in K (methane UA in TraPPE), defaults to 148.0
    :param gas_sigma: gas sigma, in Angstrom (methane UA in TraPPE), defaults to 3.73
    :param visualize: visualize the energy grid, defaults to False
    :param max_num_atoms: maximum number of atoms, defaults to 3000
    :param gaussian_height: gaussian height
    :param gaussian_width: gaussian width
    :return: energy grid
    """
    if isinstance(structure, Atoms):
        atoms = structure
    elif isinstance(structure, str):
        if Path(structure).exists():
            atoms = read(structure)
        else:
            raise FileNotFoundError(f"{structure} does not exist")
    else:
        raise TypeError("structure must be an ase Atoms object or a cif file path")

    # assert the cell lengths should be less than 2*cutoff
    cell_vectors = np.array(atoms.cell)

    # make supercell when distance between planes is less than cutoff
    plane_distances = np.linalg.norm(cell_vectors, axis=1)
    supercell = np.ceil(2 * cutoff / plane_distances).astype(int)
    atoms = atoms.repeat(supercell)
    if len(atoms) > max_num_atoms:
        raise ValueError(
            f"Too many atoms ({len(atoms)}) in the supercell,"
            f"please decrease the cutoff or increase the max_num_atoms"
        )

    # get position for grid
    if isinstance(grid_size, int):
        grid_size = np.array([grid_size] * 3)
    else:
        grid_size = eval(grid_size)
        assert len(grid_size) == 3, "grid_size must be a 3-dim vector"
    indices = np.indices(grid_size).reshape(3, -1).T
    pos_grid = indices.dot(cell_vectors / grid_size)  # (G, 3)

    # get positions for atoms
    pos_atoms = atoms.get_positions()  # (N, 3)

    # setting force field
    symbols = atoms.get_chemical_symbols()
    epsilon, sigma = get_mixing_epsilon_sigma(
        symbols, ff_type, gas_epsilon, gas_sigma
    )  # (N,) (N,)

    # calculate distance matrix
    dist_matrix = mic_distance_matrix(pos_grid, pos_atoms, cell_vectors)  # (G, N)

    # calculate energy
    if potential.lower() == "lj":
        calculated_grids = calculate_lj_potential(
            dist_matrix,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
        )  # (G,)
    elif potential.lower() == "gaussian":
        calculated_grids = calculate_gaussian(
            dist_matrix,
            height=gaussian_height,
            width=gaussian_width,
            cutoff=cutoff,
        )  # (G,)
    else:
        raise NotImplementedError(f"{potential} should be one of ['LJ', 'Gaussian']")

    if visualize:
        print("Visualizing energy grids...")
        visualize_grids(pos_grid, pos_atoms, calculated_grids)

    return calculated_grids


def cli():
    Fire(calculate_grids)
