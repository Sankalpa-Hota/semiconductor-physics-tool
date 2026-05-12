import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot

hbar = 1.0545718e-34
m0   = 9.10938356e-31
eV   = 1.60217662e-19

DARK = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,15,26,1)',
    font=dict(family='Space Mono, monospace', color='#6a8aaa', size=10),
    xaxis=dict(gridcolor='#1a2840', zerolinecolor='#1a2840', linecolor='#1a2840'),
    yaxis=dict(gridcolor='#1a2840', zerolinecolor='#1a2840', linecolor='#1a2840'),
    margin=dict(l=55, r=18, t=38, b=50),
)


def kronig_penney_dispersion(a=5e-10, V0=10*eV, b=2e-10, m=m0, num_points=500):
    E       = np.linspace(0.01 * eV, 20 * eV, num_points)
    ka_vals = []
    for Ei in E:
        try:
            alpha = np.sqrt(2 * m * Ei) / hbar
            if Ei < V0:
                beta     = np.sqrt(2 * m * (V0 - Ei)) / hbar
                cosh_val = np.cosh(np.clip(beta * b, -300, 300))
                sinh_val = np.sinh(np.clip(beta * b, -300, 300))
                # correct sign: + between the two terms
                M = (np.cos(alpha * (a - b)) * cosh_val
                     + ((beta**2 - alpha**2) / (2 * alpha * beta))
                     * np.sin(alpha * (a - b)) * sinh_val)
            else:
                gamma    = np.sqrt(2 * m * (Ei - V0)) / hbar
                cos_val  = np.cos(gamma * b)
                sin_val  = np.sin(gamma * b)
                M = (np.cos(alpha * (a - b)) * cos_val
                     - ((gamma**2 + alpha**2) / (2 * alpha * gamma))
                     * np.sin(alpha * (a - b)) * sin_val)
            ka_vals.append(float(np.real(M)))
        except Exception:
            ka_vals.append(np.nan)
    return np.array(E) / eV, np.array(ka_vals)


def plot_kronig_penney_1d(a=5e-10, V0=10*eV, b=2e-10):
    E, coska = kronig_penney_dispersion(a, V0, b)
    layout   = go.Layout(
        title=dict(text='1D Kronig-Penney: cos(ka) vs Energy',
                   font=dict(size=10, color='#2e4460'), x=0.01),
        xaxis_title='Energy (eV)',
        yaxis_title='cos(ka)',
        **DARK
    )
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(
        x=E, y=coska, mode='lines', name='cos(ka)',
        line=dict(color='#00f0ff', width=2)))
    fig.add_hline(y=1,  line=dict(color='rgba(255,62,138,0.7)',  dash='dot', width=1.5),
                  annotation_text='cos=+1',
                  annotation_font=dict(color='#ff3e8a', size=9))
    fig.add_hline(y=-1, line=dict(color='rgba(255,62,138,0.7)', dash='dot', width=1.5),
                  annotation_text='cos=-1',
                  annotation_font=dict(color='#ff3e8a', size=9))
    return plot(fig, output_type='div', include_plotlyjs=False)


def plot_kronig_penney_2d(a=5e-10, V0=10*eV, b=2e-10):
    E = np.linspace(0.01 * eV, 20 * eV, 200)
    k = np.linspace(-np.pi / a, np.pi / a, 200)
    Z = np.zeros((len(E), len(k)))
    for i, Ei in enumerate(E):
        try:
            alpha = np.sqrt(2 * m0 * Ei) / hbar
            if Ei < V0:
                beta     = np.sqrt(2 * m0 * (V0 - Ei)) / hbar
                cosh_val = np.cosh(np.clip(beta * b, -300, 300))
                sinh_val = np.sinh(np.clip(beta * b, -300, 300))
                coska    = (np.cos(alpha * (a - b)) * cosh_val
                            + ((beta**2 - alpha**2) / (2 * alpha * beta))
                            * np.sin(alpha * (a - b)) * sinh_val)
            else:
                gamma = np.sqrt(2 * m0 * (Ei - V0)) / hbar
                coska = (np.cos(alpha * (a - b)) * np.cos(gamma * b)
                         - ((gamma**2 + alpha**2) / (2 * alpha * gamma))
                         * np.sin(alpha * (a - b)) * np.sin(gamma * b))
            Z[i, :] = float(np.real(coska))
        except Exception:
            Z[i, :] = np.nan

    layout = go.Layout(
        title=dict(text='2D KP Band Structure (E-k map)',
                   font=dict(size=10, color='#2e4460'), x=0.01),
        xaxis_title='k*a/pi',
        yaxis_title='Energy (eV)',
        **DARK
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=Z, x=k * a / np.pi, y=E / eV,
            colorscale='Plasma',
            colorbar=dict(title='cos(ka)', tickfont=dict(size=8, color='#6a8aaa'))
        ),
        layout=layout
    )
    return plot(fig, output_type='div', include_plotlyjs=False)
