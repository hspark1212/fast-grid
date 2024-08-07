import warnings
from typing import Union, Iterable
from pathlib import Path

from fire import Fire

import numpy as np
from ase import Atoms
from ase.io import read
import MDAnalysis as mda

from fast_grid.ff import get_mixing_epsilon_sigma
from fast_grid.potential import lj_potential, gaussian
from fast_grid.visualize import visualize_grid

warnings.filterwarnings("ignore")


def calculate_grid(
    atoms: Union[Atoms, str],
    grid_size: Union[int, Iterable] = 30,
    grid_spacing: float = None,
    ff_type: str = "UFF",
    potential: str = "LJ",
    cutoff: float = 12.8,
    gas_epsilon: float = 148.0,  # LJ
    gas_sigma: float = 3.73,  # LJ
    visualize: bool = False,
    gaussian_height: float = 1.0,
    gaussian_width: float = 1.0,
    float16: bool = False,
    emax: float = 5000.0,
    emin: float = -5000.0,
    pallete: str = "RdBu",
    atomic_pallete_threshold: float = 0.5,
    view_atoms: bool = True,
    grid_view: bool = False,
    return_dict: bool = False,
) -> np.array:
    """Calculate the energy grid for a given structure and force field.
    It takes a structure (ase Atoms object or cif file path) and returns the energy grid.
    The supported potentials are Lennard-Jones and Gaussian.
    The supported force field is UFF.
    The gas parameters are for methane in TraPPE-UA (united-atoms) force field.
    The unit of energy is K and the unit of distance is Angstrom.
    The output shape of energy grid is grid_size * grid_size * grid_size.

    :param atoms: ase Atoms object or cif file path
    :param grid_size: grid size, for example, 30 or "(30, 30, 30)", defaults to 30
    :param grid_spacing: grid spacing, overrides grid_size, defaults to None
    :param ff_type: force field type, defaults to "UFF"
    :param potential: potential function, gaussian or lj, defaults to "LJ"
    :param cutoff: cutoff distance, defaults to 12.8
    :param gas_epsilon: gas epsilon, in K (methane UA in TraPPE), defaults to 148.0
    :param gas_sigma: gas sigma, in Angstrom (methane UA in TraPPE), defaults to 3.73
    :param visualize: visualize the energy grid, defaults to False
    :param gaussian_height: gaussian height, defaults to 1.0
    :param gaussian_width: gaussian width, defaults to 5.0
    :param float16: use float16 to save memory, defaults to False
    :param emax: clip energy values for better visualization, defaults to 5000.0
    :param emin: clip energy values for better visualization, defaults to -5000.0
    :param pallete: color pallete for visualization, defaults to "RdBu"
    :param atomic_pallete_threshold: atomic pallete threshold, defaults to 0.5
    :param view_atoms: visualize atoms, defaults to True
    :param grid_view: visualize the grid, defaults to False
    :param return_dict: return a dictionary of outputs, defaults to False
    :return: energy grid
    """
    if isinstance(atoms, Atoms):
        pass
    elif isinstance(atoms, str):
        if Path(atoms).exists():
            atoms = read(atoms)
        else:
            raise FileNotFoundError(f"{atoms} does not exist")
    else:
        raise TypeError("structure must be an ase Atoms object or a cif file path")

    if potential.lower() == "lj":
        # make supercell when distance between planes is less than cutoff * 2
        cell_volume = atoms.get_volume()
        cell_vectors = np.array(atoms.cell)
        dist_a = cell_volume / np.linalg.norm(
            np.cross(cell_vectors[1], cell_vectors[2])
        )
        dist_b = cell_volume / np.linalg.norm(
            np.cross(cell_vectors[2], cell_vectors[0])
        )
        dist_c = cell_volume / np.linalg.norm(
            np.cross(cell_vectors[0], cell_vectors[1])
        )
        plane_distances = np.array([dist_a, dist_b, dist_c])
        supercell = np.ceil(2 * cutoff / plane_distances).astype(int)
        atoms = atoms.repeat(supercell)  # make supercell
    else:
        supercell = np.array((1, 1, 1))
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

    indices = np.indices(grid_size).reshape(3, -1).T  # (G, 3)
    pos_grid = indices.dot(cell_vectors / grid_size)  # (G, 3)
    # get positions for atoms
    pos_atoms = atoms.get_positions()  # (N, 3)

    # distance matrix
    dist_matrix = mda.lib.distances.distance_array(
        pos_grid, pos_atoms, box=atoms.cell.cellpar()
    )  # (G, N)

    # calculate energy
    if potential.lower() == "lj":
        # setting force field
        symbols = atoms.get_chemical_symbols()
        epsilon, sigma = get_mixing_epsilon_sigma(
            symbols, ff_type, gas_epsilon, gas_sigma
        )  # (N,) (N,)
        calculated_grid = lj_potential(dist_matrix, epsilon, sigma, cutoff)  # (G,)

    elif potential.lower() == "gaussian":
        calculated_grid = gaussian(dist_matrix, gaussian_height, gaussian_width)  # (G,)
    else:
        raise NotImplementedError(f"{potential} should be one of ['LJ', 'Gaussian']")

    # convert to float16 to save memory
    if float16:
        # clip energy values for np.float16
        min_float16 = np.finfo(np.float16).min
        max_float16 = np.finfo(np.float16).max
        calculated_grid = np.clip(calculated_grid, min_float16, max_float16)
        # convert to float16
        calculated_grid = calculated_grid.astype(np.float16)

    if visualize:
        print(f"Visualizing energy grid | supercell {supercell}...")
        visualize_grid(
            pos_grid=indices if grid_view else pos_grid,
            atoms=atoms,
            calculated_grid=calculated_grid,
            cell_vectors=np.diag(grid_size) if grid_view else cell_vectors,
            dist_matrix=dist_matrix,
            emax=emax,
            emin=emin,
            pallete=pallete,
            atomic_pallete_threshold=atomic_pallete_threshold,
            view_atoms=view_atoms,
        )

    if return_dict:
        return {
            "atoms": atoms,  # supercelled atoms
            "supercell": supercell,
            "pos_grid": pos_grid,
            "calculated_grid": calculated_grid,
            "dist_matrix": dist_matrix,
        }

    return calculated_grid


def cli():
    Fire(calculate_grid)
