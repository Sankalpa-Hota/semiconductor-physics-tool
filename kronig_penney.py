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

# --- Safe cosh/sinh to avoid overflow ---
def cosh_safe(x):
    x = np.clip(x, -700, 700)
    return np.cosh(x)

def sinh_safe(x):
    x = np.clip(x, -700, 700)
    return np.sinh(x)

def kronig_penney_dispersion(a=5e-10, V0=10*eV, b=2e-10, m=m0, num_points=500):
    """
    Compute dispersion relation for the Kronig-Penney model.
    Returns:
        E (eV), cos(ka)
    """
    E = np.linspace(0.01*eV, 20*eV, num_points)
    cos_vals = []

    for Ei in E:
        alpha = np.sqrt(2 * m * Ei) / hbar
        if Ei < V0:
            beta = np.sqrt(2 * m * (V0 - Ei)) / hbar
            coska = np.cos(alpha*(a-b))*cosh_safe(beta*b) - ((beta**2 - alpha**2)/(2*alpha*beta))*np.sin(alpha*(a-b))*sinh_safe(beta*b)
        else:
            beta = np.sqrt(2 * m * (Ei - V0)) / hbar
            coska = np.cos(alpha*(a-b))*np.cos(beta*b) - ((beta**2 + alpha**2)/(2*alpha*beta))*np.sin(alpha*(a-b))*np.sin(beta*b)
        cos_vals.append(np.real(coska))

    return E / eV, np.array(cos_vals)

def plot_kronig_penney_1d(a=5e-10, V0=10*eV, b=2e-10):
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
    E = np.linspace(0.01*eV, 20*eV, 200)
    k = np.linspace(-np.pi/a, np.pi/a, 200)
    Z = np.zeros((len(E), len(k)))

    for i, Ei in enumerate(E):
        alpha = np.sqrt(2 * m0 * Ei) / hbar
        if Ei < V0:
            beta = np.sqrt(2 * m0 * (V0 - Ei)) / hbar
            coska = np.cos(alpha*(a-b))*cosh_safe(beta*b) - ((beta**2 - alpha**2)/(2*alpha*beta))*np.sin(alpha*(a-b))*sinh_safe(beta*b)
        else:
            beta = np.sqrt(2 * m0 * (Ei - V0)) / hbar
            coska = np.cos(alpha*(a-b))*np.cos(beta*b) - ((beta**2 + alpha**2)/(2*alpha*beta))*np.sin(alpha*(a-b))*np.sin(beta*b)
        Z[i, :] = np.real(coska)

    fig = go.Figure(data=go.Heatmap(
        z=Z, x=k*a/np.pi, y=E/eV,
        colorscale='Viridis', colorbar=dict(title="cos(ka)")
    ))
    fig.update_layout(
        title="2D Kronig–Penney Band Structure (E–k map)",
        xaxis_title="k·a/π",
        yaxis_title="Energy (eV)",
        template="plotly_dark"
    )
    return plot(fig, output_type='div', include_plotlyjs=False)
