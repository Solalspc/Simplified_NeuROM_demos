"""
pgd_plots.py
------------
Plotting utilities for the 1D element-based HiDeNN-FEM / PGD notebook.
All functions use Plotly and expect a trained PGDapprox model.

Functions
---------
plot_loss               – Flattened training loss across all modes
plot_node_displacement  – Bar chart of nodal displacement vs. reference mesh (r-adaptivity)
plot_modes              – Spatial modes u_i(x) for every trained mode
plot_gauss_displacement – u evaluated at displaced vs. initial Gauss points
plot_displacement_field – u(x) evaluated for a list of E values
plot_midpoint_vs_E      – Midpoint deflection vs E, with analytical reference
"""

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _device(model):
    """Return the device of the first parameter of *model*."""
    return next(model.parameters()).device


# ---------------------------------------------------------------------------
# 1. Training loss
# ---------------------------------------------------------------------------

def plot_loss(lossLists, width=700, height=400):
    """
    Plot the concatenated training loss over all PGD modes.

    Parameters
    ----------
    lossLists : list[list[float]]
        One sub-list per mode, as stored in results['lossLists'].
    width, height : int
        Figure dimensions in pixels.
    """
    flat = [v for mode_losses in lossLists for v in mode_losses]

    # Compute mode boundary x-positions for optional vertical lines
    boundaries = []
    cursor = 0
    for mode_losses in lossLists[:-1]:
        cursor += len(mode_losses)
        boundaries.append(cursor)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=flat,
        mode="lines",
        name="Loss",
        line=dict(color="#01426a"),
    ))

    for b in boundaries:
        fig.add_vline(x=b, line=dict(color="gray", dash="dash", width=1))

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        width=width,
        height=height,
        title="Training loss — all modes",
        xaxis=dict(title="Epoch (cumulative)", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="Loss", showgrid=True, gridcolor="lightgray"),
    )
    fig.show()


# ---------------------------------------------------------------------------
# 2. Nodal displacement (r-adaptivity)
# ---------------------------------------------------------------------------

def plot_node_displacement(model, nodes_u_ref, width=750, height=350):
    """
    Bar chart of (reference node position – trained node position) per mode.
    Meaningful only when r-adaptivity is active.

    Parameters
    ----------
    model : PGDapprox
        Trained model.
    nodes_u_ref : torch.Tensor
        Original (uniform) node positions, shape (N, 1) or (N,).
    width, height : int
        Figure dimensions in pixels.
    """
    device = _device(model)
    # Keep nodes_ref on the same device as the model for the subtraction,
    # then move to CPU for numpy conversion.
    nodes_ref = nodes_u_ref.squeeze(-1).to(device)
    node_indices = torch.arange(len(nodes_ref)).numpy()

    fig = go.Figure()
    for mode_idx in range(len(model.u_modes)):
        trained_coords = model.u_modes[mode_idx].get_coordinates().detach().squeeze(-1)
        displacement = (nodes_ref - trained_coords).cpu().numpy()

        fig.add_trace(go.Bar(
            x=node_indices,
            y=displacement,
            name=f"Mode {mode_idx + 1}",
            opacity=0.5,
        ))

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Node index",
        yaxis_title="Displacement (ref − trained)",
        width=width,
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        title="Nodal displacement per mode (r-adaptivity)",
    )
    fig.show()


# ---------------------------------------------------------------------------
# 3. Spatial modes
# ---------------------------------------------------------------------------

def plot_modes(model, n_points=200, width=700, height=400):
    """
    Plot each spatial mode u_i(x) over its own coordinate range.

    Parameters
    ----------
    model : PGDapprox
        Trained model (switched to eval internally, restored to train on exit).
    n_points : int
        Number of evaluation points per mode.
    width, height : int
        Figure dimensions in pixels.
    """
    model.eval()
    fig = go.Figure()

    with torch.no_grad():
        for i in range(model.current_index):
            coords = model.u_modes[i].get_coordinates().reshape(-1)
            # linspace lives on the same device as the mode's coordinates
            x_plot = torch.linspace(coords.min(), coords.max(), n_points,
                                    device=coords.device)
            u_plot = model.u_modes[i](x_plot).reshape(-1)
            fig.add_trace(go.Scatter(
                x=x_plot.cpu().numpy(),
                y=u_plot.cpu().numpy(),
                mode="lines",
                name=f"Mode {i + 1}",
            ))

    fig.update_layout(
        title="PGD spatial modes",
        xaxis_title="x",
        yaxis_title="u",
        legend_title="Modes",
        width=width,
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.show()
    model.train()


# ---------------------------------------------------------------------------
# 4. Gauss-point displacement trick
# ---------------------------------------------------------------------------

def plot_gauss_displacement(model, x_gauss_ini, E=500.0, width=700, height=400):
    """
    Evaluate the PGD solution at the *trained* Gauss points (displaced by
    r-adaptivity) and at the *initial* Gauss points, then overlay both.

    Parameters
    ----------
    model : PGDapprox
        Trained model (must be in train mode before the call; toggled internally).
    x_gauss_ini : torch.Tensor
        Initial Gauss-point positions saved at the start of training
        (results['x_gauss_ini']).
    E : float
        Young's modulus at which to evaluate the solution.
    width, height : int
        Figure dimensions in pixels.
    """
    device = _device(model)

    # Move the initial Gauss points to the model's device
    x_gauss_ini = x_gauss_ini.to(device)

    # Fetch current Gauss points from a forward pass in train mode
    model.train()
    with torch.no_grad():
        _, _, L_x_g, _, _, _, _ = model()

    fig = go.Figure()

    model.eval()
    # Create E tensor on the same device as the model
    E_tensor = torch.tensor([E], dtype=torch.float32, device=device)

    for mode_idx in range(model.current_index):
        x_gauss_disp = L_x_g[mode_idx].squeeze(-1).detach()

        with torch.no_grad():
            u_g = model(x_gauss_disp, E_tensor)

        y = u_g[0].detach().cpu().numpy().ravel()
        fig.add_trace(go.Scatter(
            x=x_gauss_disp.cpu().numpy(),
            y=y,
            mode="markers",
            name=f"Mode {mode_idx + 1} — displaced pts",
        ))

    # Initial Gauss points (shown once, common reference)
    with torch.no_grad():
        u_ini = model(x_gauss_ini, E_tensor)
    y_ini = u_ini[0].detach().cpu().numpy().ravel()
    fig.add_trace(go.Scatter(
        x=x_gauss_ini.cpu().numpy(),
        y=y_ini,
        mode="markers",
        marker=dict(symbol="cross", size=8),
        name="Initial Gauss pts",
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        width=width,
        height=height,
        title=f"Solution at Gauss points — E = {E}",
        xaxis=dict(title="x [mm]", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="u(x) [mm]", showgrid=True, gridcolor="lightgray"),
        legend=dict(x=0, y=1),
    )
    fig.show()
    model.train()


# ---------------------------------------------------------------------------
# 5. Displacement field for a list of E values
# ---------------------------------------------------------------------------

def plot_displacement_field(model, E_values, x_range=(0.0, 6.28), n_points=50,
                            width=700, height=400):
    """
    Evaluate and plot u(x) for each Young's modulus value in *E_values*.

    Parameters
    ----------
    model : PGDapprox
        Trained model.
    E_values : list[float] | torch.Tensor
        Young's modulus values to evaluate.
    x_range : tuple[float, float]
        (x_min, x_max) evaluation interval.
    n_points : int
        Number of spatial evaluation points.
    width, height : int
        Figure dimensions in pixels.
    """
    device = _device(model)

    if not isinstance(E_values, torch.Tensor):
        E_values = torch.tensor(E_values, dtype=torch.float32)
    E_values = E_values.to(device)

    x_test = torch.linspace(x_range[0], x_range[1], n_points, device=device)

    model.eval()
    with torch.no_grad():
        u_list = model(x_test, E_values)

    fig = go.Figure()
    for i, u in enumerate(u_list):
        y = u.detach().cpu().numpy().ravel()
        fig.add_trace(go.Scatter(
            x=x_test.cpu().numpy(),
            y=y,
            mode="markers",
            name=f"E = {E_values[i].item():.4g}",
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        width=width,
        height=height,
        title="Displacement field u(x)",
        xaxis=dict(title="x [mm]", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="u(x) [mm]", showgrid=True, gridcolor="lightgray"),
        legend=dict(x=0, y=1),
    )
    fig.show()
    model.train()


# ---------------------------------------------------------------------------
# 6. Midpoint deflection vs E  (with analytical reference)
# ---------------------------------------------------------------------------

def plot_midpoint_vs_E(model, x_mid=3.14, E_range=(1e2, 1e3), n_E=100,
                       q=1000.0, L=6.28, width=700, height=400):
    """
    Plot the midpoint displacement u(x_mid, E) vs E alongside the analytical
    solution for a uniformly loaded bar: u = q * L² / (8 * E).

    Parameters
    ----------
    model : PGDapprox
        Trained model.
    x_mid : float
        Spatial coordinate of the midpoint (default π ≈ 3.14 for L = 2π).
    E_range : tuple[float, float]
        (E_min, E_max) range for the parameter sweep.
    n_E : int
        Number of E values to sample.
    q : float
        Distributed load amplitude (must match the f() used during training).
    L : float
        Beam length (must match the mesh used during training).
    width, height : int
        Figure dimensions in pixels.
    """
    device = _device(model)

    x_midpoint = torch.tensor([x_mid], dtype=torch.float32, device=device)
    E_test = torch.linspace(E_range[0], E_range[1], n_E, device=device)

    model.eval()
    with torch.no_grad():
        u_pgd = model(x_midpoint, E_test)

    y_pgd = [u.detach().cpu().numpy()[0] for u in u_pgd]

    E_np = E_test.cpu().numpy()
    u_analytical = (q * L**2) / (8.0 * E_np)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=E_np,
        y=y_pgd,
        mode="markers",
        name="PGD model",
    ))
    fig.add_trace(go.Scatter(
        x=E_np,
        y=u_analytical,
        mode="lines",
        name="Analytical u(L/2)",
        line=dict(color="red", dash="dash"),
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        width=width,
        height=height,
        title=f"Midpoint deflection u(x={x_mid}) vs E",
        xaxis=dict(title="E [MPa]", showgrid=True, gridcolor="lightgray"),
        yaxis=dict(title="u(mid) [mm]", showgrid=True, gridcolor="lightgray"),
        legend=dict(x=0, y=1),
    )
    fig.show()
    model.train()


# ---------------------------------------------------------------------------
# 7. Error compared to the no-r-adapt reference run
# ---------------------------------------------------------------------------

def compute_error_vs_ref(model, model_ref,
                         x_range=(0.0, 6.28), E_range=(1e2, 1e3),
                         n_x=100, n_E=50,
                         plot=True, width=800, height=600):
    """
    Evaluate both *model* and *model_ref* on a uniform (x, E) grid and
    return pointwise error metrics, plus a unified dashboard plot.

    Parameters
    ----------
    model : PGDapprox
        Model to evaluate (e.g., r-adaptivity run).
    model_ref : PGDapprox
        Reference model (e.g., no r-adaptivity run).
    x_range : tuple[float, float]
        Spatial evaluation interval.
    E_range : tuple[float, float]
        Parameter evaluation interval.
    n_x, n_E : int
        Number of evaluation points.
    plot : bool
        If True, renders a unified 2x2 marginal Plotly dashboard.
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    metrics : dict
        Scalar error indicators.
    U, U_ref : torch.Tensor
        Evaluated solutions, shape (n_E, n_x).
    """
    # 1. Save original training states
    was_training = model.training
    was_training_ref = model_ref.training

    model.eval()
    model_ref.eval()

    # Each model may live on a different device; generate grids per model.
    device     = _device(model)
    device_ref = _device(model_ref)

    # 2. Generate grids — one per device
    x_grid     = torch.linspace(x_range[0], x_range[1], n_x, device=device)
    E_grid     = torch.linspace(E_range[0], E_range[1], n_E, device=device)
    x_grid_ref = torch.linspace(x_range[0], x_range[1], n_x, device=device_ref)
    E_grid_ref = torch.linspace(E_range[0], E_range[1], n_E, device=device_ref)

    with torch.no_grad():
        u_list     = model(x_grid, E_grid)
        u_ref_list = model_ref(x_grid_ref, E_grid_ref)

    # Stack into matrices (n_E, n_x) — bring everything to CPU for comparison
    U     = torch.stack([u.detach().cpu().reshape(-1) for u in u_list])
    U_ref = torch.stack([u.detach().cpu().reshape(-1) for u in u_ref_list])

    # 3. Robust Error Calculation
    abs_err = (U - U_ref).abs()

    # Dynamic clamping prevents artificial division-by-zero spikes at boundaries
    # where the analytical solution is exactly 0.
    eps = 1e-8 * U_ref.abs().max().clamp(min=1e-8)
    rel_err = abs_err / (U_ref.abs().clamp(min=eps))

    metrics = {
        "relative_norm":       (torch.norm(U - U_ref) / torch.norm(U_ref)).item(),
        "mean_relative_error": rel_err.mean().item(),
        "max_relative_error":  rel_err.max().item(),
        "l2_absolute":         torch.sqrt((abs_err ** 2).mean()).item(),
    }

    print("=== Error vs. reference ===")
    for k, v in metrics.items():
        print(f"  {k:<26s}: {v:.4e}")

    # 4. Unified Subplot Dashboard
    if plot:
        # All tensors are already on CPU after the stack above
        x_np   = x_grid.cpu().numpy()
        E_np   = E_grid.cpu().numpy()
        rel_np = rel_err.numpy()

        mean_over_E = rel_err.mean(dim=0).numpy()  # Shape (n_x,) - Marginal X
        mean_over_x = rel_err.mean(dim=1).numpy()  # Shape (n_E,) - Marginal Y

        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.75, 0.25],
            row_heights=[0.75, 0.25],
            horizontal_spacing=0.05,
            vertical_spacing=0.08,
            shared_xaxes=True,
            shared_yaxes=True,
        )

        # Top-Left: Heatmap
        fig.add_trace(go.Heatmap(
            z=rel_np, x=x_np, y=E_np,
            colorscale="Reds",
            colorbar=dict(title="Rel Error", len=0.75, y=0.625),
            name="Heatmap"
        ), row=1, col=1)

        # Top-Right: Mean Error vs E (rotated to align with Y axis)
        fig.add_trace(go.Scatter(
            x=mean_over_x, y=E_np,
            mode="lines",
            line=dict(color="#c0392b", width=2),
            name="Mean over x",
            showlegend=False
        ), row=1, col=2)

        # Bottom-Left: Mean Error vs X (aligned with X axis)
        fig.add_trace(go.Scatter(
            x=x_np, y=mean_over_E,
            mode="lines",
            line=dict(color="#01426a", width=2),
            name="Mean over E",
            showlegend=False
        ), row=2, col=1)

        # Layout adjustments
        fig.update_layout(
            title="Pointwise Relative Error Profile: |u − u_ref| / |u_ref|",
            plot_bgcolor="rgba(0,0,0,0)",
            width=width,
            height=height,
            margin=dict(l=50, r=20, t=60, b=50),
        )

        fig.update_yaxes(title_text="E [MPa]", row=1, col=1)
        fig.update_xaxes(title_text="x [mm]", row=2, col=1)
        fig.update_xaxes(title_text="Avg Error", row=1, col=2,
                         showgrid=True, gridcolor="lightgray")
        fig.update_yaxes(title_text="Avg Error", row=2, col=1,
                         showgrid=True, gridcolor="lightgray", autorange="reversed")

        fig.show()

    # 5. Safe state restoration
    if was_training:
        model.train()
    if was_training_ref:
        model_ref.train()

    return metrics, U, U_ref