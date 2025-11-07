import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot

hbar = 1.0545718e-34
m0 = 9.10938356e-31
eV = 1.60217662e-19

def kronig_penney_dispersion(a=5e-10, V0=10*eV, b=2e-10, m=m0, num_points=500):
    E = np.linspace(0.01*eV, 20*eV, num_points)
    ka_vals = []

    for Ei in E:
        try:
            alpha = np.sqrt(2*m*Ei)/hbar
            if Ei < V0:
                beta = np.sqrt(2*m*(V0-Ei))/hbar
                cosh_beta_b = np.cosh(np.clip(beta*b, -100, 100))
                sinh_beta_b = np.sinh(np.clip(beta*b, -100, 100))
            else:
                beta = 1j*np.sqrt(2*m*(Ei-V0))/hbar
                cosh_beta_b = np.cosh(beta*b)
                sinh_beta_b = np.sinh(beta*b)

            M = np.cos(alpha*(a-b))*cosh_beta_b - ((beta**2 - alpha**2)/(2*alpha*beta))*np.sin(alpha*(a-b))*sinh_beta_b
            ka_vals.append(np.real(M))
        except:
            ka_vals.append(np.nan)

    return np.array(E)/eV, np.array(ka_vals)

def plot_kronig_penney_1d(a=5e-10, V0=10*eV, b=2e-10):
    E, coska = kronig_penney_dispersion(a, V0, b)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=E, y=coska, mode='lines', name='cos(ka)'))
    fig.add_hline(y=1, line_dash="dot", line_color="red")
    fig.add_hline(y=-1, line_dash="dot", line_color="red")
    fig.update_layout(
        title="1D Kronig–Penney: cos(ka) vs Energy",
        xaxis_title="Energy (eV)", yaxis_title="cos(ka)",
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return plot(fig, output_type='div', include_plotlyjs=False)

def plot_kronig_penney_2d(a=5e-10, V0=10*eV, b=2e-10):
    E = np.linspace(0.01*eV, 20*eV, 200)
    k = np.linspace(-np.pi/a, np.pi/a, 200)
    Z = np.zeros((len(E), len(k)))

    for i, Ei in enumerate(E):
        try:
            alpha = np.sqrt(2*m0*Ei)/hbar
            beta = np.sqrt(2*m0*abs(V0-Ei))/hbar
            coska = np.cos(alpha*(a-b))*np.cosh(np.clip(beta*b, -100,100)) - ((beta**2 - alpha**2)/(2*alpha*beta))*np.sin(alpha*(a-b))*np.sinh(np.clip(beta*b, -100,100))
            Z[i, :] = np.real(coska)
        except:
            Z[i, :] = np.nan

    fig = go.Figure(data=go.Heatmap(
        z=Z, x=k*a/np.pi, y=E/eV,
        colorscale='Viridis', colorbar=dict(title="cos(ka)")
    ))
    fig.update_layout(
        title="2D Kronig–Penney Band Structure (E–k map)",
        xaxis_title="k·a/π", yaxis_title="Energy (eV)",
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return plot(fig, output_type='div', include_plotlyjs=False)
