import numpy as np
from ase import Atoms
from ase.data.colors import jmol_colors

import plotly.graph_objects as go


def visualize_grid(
    pos_grid: np.array,
    atoms: Atoms,
    calculated_grid: np.array,
    dist_matrix: np.array = None,
    emax: float = 5000,
    emin: float = -5000,
    pallete: str = "RdBu",
):
    pos_atoms = atoms.get_positions()
    cell_vectors = np.array(atoms.cell)
    # clip energy values for better visualization
    calculated_grid = np.clip(calculated_grid, emin, emax)

    # Create a custom colorscale for energy values
    if pallete == "transparent":
        color = calculated_grid
        colorscale = [
            [0.0, "rgba(0, 0, 255, 0)"],  # Transparent for low values
            [0.5, "rgba(0, 0, 255, 0)"],  # Still transparent in the middle
            [1.0, "rgba(0, 0, 255, 1.0)"],
        ]
    elif pallete == "atomic":
        cloest_atom = np.argmin(dist_matrix, axis=1)  # (G,)
        cloest_atom_types = atoms.numbers[cloest_atom]  # (G,)
        rgb_atom_colors = jmol_colors[cloest_atom_types] * 255  # (G, 3)
        calculated_grid[calculated_grid < 0.5] = 0
        rgba_atom_colors = np.concatenate(
            [rgb_atom_colors, calculated_grid[:, None]], axis=1
        )  # (G, 4)
        rgba_atom_colors = ["rgba" + str(tuple(color)) for color in rgba_atom_colors]
        color = rgba_atom_colors
        colorscale = None
    else:
        color = calculated_grid
        colorscale = pallete

    # Create figure with subplots
    fig = go.Figure()

    # Add a plot for cell
    a, b, c = cell_vectors
    lines = [
        [[0, 0, 0], a],
        [[0, 0, 0], b],
        [[0, 0, 0], c],
        [a, a + b],
        [a, a + c],
        [b, b + a],
        [b, b + c],
        [c, c + a],
        [c, c + b],
        [a + b, a + b + c],
        [a + c, a + c + b],
        [b + c, b + c + a],
    ]
    line_traces = []
    for line in lines:
        x_values, y_values, z_values = zip(*line)
        line_trace = go.Scatter3d(
            x=x_values,
            y=y_values,
            z=z_values,
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
        )
        line_traces.append(line_trace)
    fig.add_traces(line_traces)

    # Add a plot for energy grid points
    fig.add_trace(
        go.Scatter3d(
            x=pos_grid[:, 0],
            y=pos_grid[:, 1],
            z=pos_grid[:, 2],
            mode="markers",
            hovertemplate=(
                "Energy: %{marker.color:.2f} "
                "<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})"
            ),
            marker=dict(
                size=6,
                color=color,
                colorscale=colorscale,
                opacity=0.3,
                colorbar=dict(
                    thickness=20,
                    title="Energy",
                ),
                # cmin=emin,
                # cmax=emax,
            ),
            showlegend=False,
        ),
    )

    # Add a plot for atoms
    fig.add_trace(
        go.Scatter3d(
            x=pos_atoms[:, 0],
            y=pos_atoms[:, 1],
            z=pos_atoms[:, 2],
            mode="markers",
            hovertemplate="Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})",
            marker=dict(
                size=10,
                color=[
                    "rgb" + str(tuple(jmol_colors[atom] * 255))
                    for atom in atoms.numbers
                ],
            ),
            showlegend=False,
        ),
    )

    # update layout
    # Customize the layout with a background theme

    fig.update_layout(
        title="3D Scatter Plot of Grid Points",
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    fig.show()
