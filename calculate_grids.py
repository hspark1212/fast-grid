from typing import Union, Iterable
from pathlib import Path

from fire import Fire

import numpy as np
from ase import Atoms
from ase.io import read

from fast_grid.ff import get_mixing_epsilon_sigma
from fast_grid.utils import mic_distance_matrix_triclinic
from fast_grid.potential import calculate_lj_potential, calculate_gaussian
from fast_grid.visualize import visualize_grids

# TODO: add requirements.txt


def calculate_grids(
    structure: Union[Atoms, str],
    grid_size: Union[int, Iterable] = 30,
    ff_type: str = "UFF",
    potential: str = "LJ",
    cutoff: float = 12.8,
    gas_epsilon: float = 148.0,
    gas_sigma: float = 3.73,
    visualize: bool = False,
):
    """_summary_
    Calculate the energy grid for a given structure and force field.
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
    cell_lengths = np.linalg.norm(cell_vectors, axis=1)

    if np.any(cell_lengths < 2 * cutoff):
        # TODO: update making supercell
        raise ValueError("cell length is less than 2 * cutoff")

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
    dist_matrix = mic_distance_matrix_triclinic(
        pos_grid, pos_atoms, cell_vectors
    )  # (G, N)

    # calculate energy
    if potential == "LJ":
        calculated_grids = calculate_lj_potential(
            dist_matrix,
            epsilon=epsilon,
            sigma=sigma,
            cutoff=cutoff,
        )  # (G,)
    elif potential == "Gaussian":
        calculated_grids = calculate_gaussian(
            dist_matrix,
            center=0.0,  # TODO: check the center
            amplitude=1.0,
            width=100,
        )  # (G,)
    else:
        raise NotImplementedError(f"{potential} should be one of ['LJ', 'Gaussian']")

    if visualize:
        print("Visualizing energy grids...")
        visualize_grids(pos_grid, pos_atoms, calculated_grids)

    return calculated_grids


if __name__ == "__main__":
    Fire(calculate_grids)
