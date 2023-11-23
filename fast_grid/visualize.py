import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_grid(
    pos_grid: np.array,
    pos_atoms: np.array,
    calculated_grid: np.array,
    emax: float = 5000,
    emin: float = -5000,
):
    # clip energy values for better visualization
    calculated_grid = np.clip(calculated_grid, emin, emax)

    # Create figure with subplots
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])

    # Add a plot for energy grid points
    fig.add_trace(
        go.Scatter3d(
            x=pos_grid[:, 0],
            y=pos_grid[:, 1],
            z=pos_grid[:, 2],
            mode="markers",
            hovertemplate=(
                "Energy: %{marker.color:.2f} <br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})"
            ),
            marker=dict(
                size=6,
                color=calculated_grid,
                colorscale="RdBu",
                opacity=0.9,
                colorbar=dict(
                    thickness=20,
                    title="Energy",
                ),
                # cmin=emin,
                # cmax=emax,
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Add a plot for atoms
    fig.add_trace(
        go.Scatter3d(
            x=pos_atoms[:, 0],
            y=pos_atoms[:, 1],
            z=pos_atoms[:, 2],
            mode="markers",
            hovertemplate="Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})",
            marker=dict(size=6, color="rgba(0, 0, 0, 0.5)"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # update layout
    fig.update_layout(
        title="3D Scatter Plot of Grid Points",
        scene=dict(
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            zaxis_title="Z-axis",
            bgcolor="white",
        ),
        scene_aspectmode="cube",  # Maintain aspect ratio for better spatial understanding
    )

    # Add interactive features
    fig.update_layout(
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    )  # Adjust camera for initial view

    fig.show()
