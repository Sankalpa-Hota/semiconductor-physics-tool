# reciprocal_lattice.py
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot

# Default: four coplanar points in the xy plane (arbitrary non-square quad)
DEFAULT_VERTICES = np.array([
    [0.0, 0.0, 0.0],
    [1.2, 0.15, 0.0],
    [0.4, 1.05, 0.0],
    [0.9, 0.75, 0.0],
])


def reciprocal_lattice(a1, a2, a3):
    V = np.dot(a1, np.cross(a2, a3))
    if abs(V) < 1e-18:
        raise ValueError('Primitive vectors are coplanar or degenerate (zero cell volume).')
    b1 = 2 * np.pi * np.cross(a2, a3) / V
    b2 = 2 * np.pi * np.cross(a3, a1) / V
    b3 = 2 * np.pi * np.cross(a1, a2) / V
    return b1, b2, b3


def vertices_to_primitive(vertices, c_axis=0.6):
    """
    vertices: (N, 3) with N >= 3. Vertex 0 is the origin corner R0.
    a1 = R1 - R0, a2 = R2 - R0. If R3-R0 is linearly independent of (a1,a2,a3 spans 3D),
    use it as a3; otherwise a3 = c_axis * n̂ with n̂ = unit(a1×a2).
    """
    V = np.asarray(vertices, dtype=float)
    if V.shape[0] < 3 or V.shape[1] != 3:
        raise ValueError('Need at least 3 vertices as (x,y,z).')
    R0 = V[0]
    a1 = V[1] - R0
    a2 = V[2] - R0
    cr = np.cross(a1, a2)
    nrm = np.linalg.norm(cr)

    a3 = None
    if V.shape[0] >= 4:
        a_try = V[3] - R0
        vol = abs(np.dot(a_try, cr))
        scale = max(
            np.linalg.norm(a1) * np.linalg.norm(a2) * np.linalg.norm(a_try),
            1e-15,
        )
        if vol > 1e-10 * scale:
            a3 = a_try

    if a3 is None:
        if nrm < 1e-14:
            a3 = np.array([0.0, 0.0, float(c_axis)], dtype=float)
        else:
            a3 = cr / nrm * float(c_axis)

    return a1, a2, a3, R0


def _parallelepiped_edges(R0, a1, a2, a3):
    """8 corners and 12 edges from R0 + linear combinations of a1,a2,a3 in {0,1}."""
    c = []
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                c.append(R0 + i * a1 + j * a2 + k * a3)
    c = np.array(c)
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    ]
    return c, edges


def plot_reciprocal_lattice_3d(vertices=None, c_axis=0.6):
    """
    Plot user vertices in 3D, primitive vectors from vertex 0, and reciprocal b1,b2,b3 from origin.
    """
    if vertices is None:
        vertices = DEFAULT_VERTICES.copy()

    verts = np.asarray(vertices, dtype=float)
    try:
        a1, a2, a3, R0 = vertices_to_primitive(verts, c_axis=c_axis)
        b1, b2, b3 = reciprocal_lattice(a1, a2, a3)
    except ValueError as e:
        return (
            "<p style='color:#ff3e8a;padding:1rem;font-family:monospace;font-size:0.75rem'>"
            f"Lattice error: {e}</p>"
        )

    DARK_SCENE = dict(
        xaxis=dict(backgroundcolor='rgba(10,15,26,1)', gridcolor='#1a2840',
                   zerolinecolor='#1a2840', tickfont=dict(color='#6a8aaa', size=8)),
        yaxis=dict(backgroundcolor='rgba(10,15,26,1)', gridcolor='#1a2840',
                   zerolinecolor='#1a2840', tickfont=dict(color='#6a8aaa', size=8)),
        zaxis=dict(backgroundcolor='rgba(10,15,26,1)', gridcolor='#1a2840',
                   zerolinecolor='#1a2840', tickfont=dict(color='#6a8aaa', size=8)),
        bgcolor='rgba(10,15,26,1)',
        aspectmode='data',
    )

    fig = go.Figure()

    # User-specified vertices (markers + labels)
    names = [f'R{i}' for i in range(len(verts))]
    fig.add_trace(go.Scatter3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        mode='markers+text',
        marker=dict(size=6, color='#ffffff', line=dict(color='#00f0ff', width=2)),
        text=names,
        textposition='top center',
        textfont=dict(color='#00f0ff', size=9),
        name='Your vertices',
        hovertemplate='%{text}<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>',
    ))

    # Primitive cell wireframe (from R0)
    corners, ed_ix = _parallelepiped_edges(R0, a1, a2, a3)
    for i0, i1 in ed_ix:
        p0, p1 = corners[i0], corners[i1]
        fig.add_trace(go.Scatter3d(
            x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
            mode='lines',
            line=dict(color='#6a8aaa', width=2),
            showlegend=False,
            hoverinfo='skip',
        ))

    real_vecs = [a1, a2, a3]
    real_cols = ['#00f0ff', '#00ff9d', '#ffb800']
    real_names = ['a1', 'a2', 'a3']

    for vec, col, nm in zip(real_vecs, real_cols, real_names):
        p1 = R0 + vec
        fig.add_trace(go.Scatter3d(
            x=[R0[0], p1[0]], y=[R0[1], p1[1]], z=[R0[2], p1[2]],
            mode='lines+markers+text',
            line=dict(color=col, width=5),
            marker=dict(size=4, color=col),
            text=['', nm],
            textposition='top center',
            textfont=dict(color=col, size=10),
            name=nm + ' (direct)',
        ))

    recip_vecs = [b1, b2, b3]
    recip_cols = ['#ff3e8a', '#9b5de5', '#4cc9f0']
    recip_names = ['b1', 'b2', 'b3']

    for vec, col, nm in zip(recip_vecs, recip_cols, recip_names):
        p1 = R0 + vec * 0.35
        fig.add_trace(go.Scatter3d(
            x=[R0[0], p1[0]], y=[R0[1], p1[1]], z=[R0[2], p1[2]],
            mode='lines+markers+text',
            line=dict(color=col, width=5, dash='dash'),
            marker=dict(size=4, color=col),
            text=['', nm + '×0.35'],
            textposition='top center',
            textfont=dict(color=col, size=10),
            name=nm + ' (recip., scaled)',
        ))

    fig.update_layout(
        scene=DARK_SCENE,
        title=dict(
            text='Real-space vertices, primitive cell, and reciprocal vectors',
            font=dict(size=10, color='#2e4460', family='Space Mono, monospace'),
            x=0.01),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Space Mono, monospace', color='#6a8aaa', size=10),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1a2840',
                    borderwidth=1, font=dict(size=9)),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return plot(fig, output_type='div', include_plotlyjs=False)


if __name__ == '__main__':
    a1 = np.array([1.0, 0.0, 0.0])
    a2 = np.array([0.0, 1.0, 0.0])
    a3 = np.array([0.0, 0.0, 1.0])
    b1, b2, b3 = reciprocal_lattice(a1, a2, a3)
    print('b1', b1)
