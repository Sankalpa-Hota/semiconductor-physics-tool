# reciprocal_lattice.py
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot


def reciprocal_lattice(a1, a2, a3):
    V  = np.dot(a1, np.cross(a2, a3))
    b1 = 2 * np.pi * np.cross(a2, a3) / V
    b2 = 2 * np.pi * np.cross(a3, a1) / V
    b3 = 2 * np.pi * np.cross(a1, a2) / V
    return b1, b2, b3


def plot_reciprocal_lattice_3d(a1=None, a2=None, a3=None):
    if a1 is None:
        a1 = np.array([1.0, 0.0, 0.0])
        a2 = np.array([0.0, 1.0, 0.0])
        a3 = np.array([0.0, 0.0, 1.0])

    b1, b2, b3 = reciprocal_lattice(a1, a2, a3)

    DARK_SCENE = dict(
        xaxis=dict(backgroundcolor='rgba(10,15,26,1)', gridcolor='#1a2840',
                   zerolinecolor='#1a2840', tickfont=dict(color='#6a8aaa', size=8)),
        yaxis=dict(backgroundcolor='rgba(10,15,26,1)', gridcolor='#1a2840',
                   zerolinecolor='#1a2840', tickfont=dict(color='#6a8aaa', size=8)),
        zaxis=dict(backgroundcolor='rgba(10,15,26,1)', gridcolor='#1a2840',
                   zerolinecolor='#1a2840', tickfont=dict(color='#6a8aaa', size=8)),
        bgcolor='rgba(10,15,26,1)',
        aspectmode='cube',
    )

    fig = go.Figure()

    real_vecs  = [a1, a2, a3]
    real_cols  = ['#00f0ff', '#00ff9d', '#ffb800']
    real_names = ['a1', 'a2', 'a3']

    recip_vecs  = [b1, b2, b3]
    recip_cols  = ['#ff3e8a', '#9b5de5', '#4cc9f0']
    recip_names = ['b1', 'b2', 'b3']

    for vec, col, nm in zip(real_vecs, real_cols, real_names):
        fig.add_trace(go.Scatter3d(
            x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
            mode='lines+markers+text',
            line=dict(color=col, width=5),
            marker=dict(size=4, color=col),
            text=['', nm],
            textposition='top center',
            textfont=dict(color=col, size=10),
            name=nm
        ))

    for vec, col, nm in zip(recip_vecs, recip_cols, recip_names):
        fig.add_trace(go.Scatter3d(
            x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
            mode='lines+markers+text',
            line=dict(color=col, width=5, dash='dash'),
            marker=dict(size=4, color=col),
            text=['', nm],
            textposition='top center',
            textfont=dict(color=col, size=10),
            name=nm
        ))

    fig.update_layout(
        scene=DARK_SCENE,
        title=dict(text='Real & Reciprocal Lattice (3D)',
                   font=dict(size=10, color='#2e4460', family='Space Mono, monospace'),
                   x=0.01),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Space Mono, monospace', color='#6a8aaa', size=10),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1a2840',
                    borderwidth=1, font=dict(size=9)),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return plot(fig, output_type='div', include_plotlyjs=False)


if __name__ == "__main__":
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.0, 1.0, 0.0])
    a3 = np.array([0.0, 0.0, 1.0])
    b1, b2, b3 = reciprocal_lattice(a1, a2, a3)
    print(f"b1 = {b1}")
    print(f"b2 = {b2}")
    print(f"b3 = {b3}")

