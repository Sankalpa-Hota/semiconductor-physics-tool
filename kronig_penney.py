
# kronig_penney.py
"""
Kronig-Penney Model Visualization and Computation
Author: Sankalpa Hota
Date: 2025
Description:
    Implements the 1D Kronig–Penney model for electron band structure visualization.
    Includes:
        - Calculation of allowed energy bands
        - 1D band dispersion plot
        - 2D E–k relationship heatmap
"""

import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot

hbar = 1.0545718e-34  # J.s
m0 = 9.10938356e-31   # electron rest mass
eV = 1.60217662e-19   # J/eV

def kronig_penney_dispersion(a=5e-10, V0=10*eV, b=2e-10, m=m0, num_points=500):
    """
    Compute dispersion relation for the Kronig-Penney model.
    Parameters:
        a (m): lattice period
        V0 (J): barrier height
        b (m): barrier width
        m (kg): electron mass
        num_points: resolution for energy range
    Returns:
        k (1/m), E (J), cos(ka) array for allowed/forbidden bands
    """

    # Energy range
    E = np.linspace(0.01 * eV, 20 * eV, num_points)
    ka_vals = []
    cos_vals = []

    for Ei in E:
        alpha = np.sqrt(2 * m * Ei) / hbar
        beta = np.sqrt(2 * m * (V0 - Ei)) / hbar if Ei < V0 else 1j * np.sqrt(2 * m * (Ei - V0)) / hbar
        M = np.cos(alpha * (a - b)) * np.cosh(beta * b) - \
            ((beta**2 - alpha**2) / (2 * alpha * beta)) * np.sin(alpha * (a - b)) * np.sinh(beta * b)
        ka_vals.append(M.real)
        cos_vals.append(M.real)

    return np.array(E) / eV, np.array(ka_vals)

def plot_kronig_penney_1d(a=5e-10, V0=10*eV, b=2e-10):
    """
    Generates a 1D Kronig-Penney dispersion plot.
    """
    E, coska = kronig_penney_dispersion(a, V0, b)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=E, y=coska, mode='lines', name='cos(ka)'))
    fig.add_hline(y=1, line_dash="dot", line_color="red")
    fig.add_hline(y=-1, line_dash="dot", line_color="red")

    fig.update_layout(
        title="1D Kronig–Penney Model: cos(ka) vs Energy",
        xaxis_title="Energy (eV)",
        yaxis_title="cos(ka)",
        template="plotly_dark"
    )
    return plot(fig, output_type='div', include_plotlyjs=False)

def plot_kronig_penney_2d(a=5e-10, V0=10*eV, b=2e-10):
    """
    Generates a 2D Kronig-Penney band diagram (E vs k map).
    """
    E = np.linspace(0.01 * eV, 20 * eV, 400)
    k = np.linspace(-np.pi/a, np.pi/a, 400)
    Z = np.zeros((len(E), len(k)))

    for i, Ei in enumerate(E):
        alpha = np.sqrt(2 * m0 * Ei) / hbar
        beta = np.sqrt(2 * m0 * (V0 - Ei)) / hbar if Ei < V0 else 1j * np.sqrt(2 * m0 * (Ei - V0)) / hbar
        coska = np.cos(alpha * (a - b)) * np.cosh(beta * b) - \
                ((beta**2 - alpha**2) / (2 * alpha * beta)) * np.sin(alpha * (a - b)) * np.sinh(beta * b)
        Z[i, :] = np.real(coska)

    fig = go.Figure(data=go.Heatmap(
        z=Z, x=k * a / np.pi, y=E / eV,
        colorscale='Viridis', colorbar=dict(title="cos(ka)")
    ))
    fig.update_layout(
        title="2D Kronig–Penney Band Structure (E–k map)",
        xaxis_title="k·a/π",
        yaxis_title="Energy (eV)",
        template="plotly_dark"
    )
    return plot(fig, output_type='div', include_plotlyjs=False)
