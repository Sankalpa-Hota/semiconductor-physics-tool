# reciprocal_lattice.py
import numpy as np
import plotly.graph_objects as go

def reciprocal_lattice(a1, a2, a3):
    """
    Compute reciprocal lattice vectors
    b1 = 2*pi * (a2 x a3) / (a1 . (a2 x a3))
    """
    volume = np.dot(a1, np.cross(a2, a3))
    b1 = 2*np.pi * np.cross(a2, a3)/volume
    b2 = 2*np.pi * np.cross(a3, a1)/volume
    b3 = 2*np.pi * np.cross(a1, a2)/volume
    return b1, b2, b3

def plot_lattice(a1, a2, a3, b1, b2, b3):
    fig = go.Figure()

    # Real lattice vectors
    origin = np.array([0,0,0])
    for vec, color in zip([a1,a2,a3], ['red','green','blue']):
        fig.add_trace(go.Scatter3d(x=[0,vec[0]], y=[0,vec[1]], z=[0,vec[2]],
                                   mode='lines+markers', marker=dict(size=4),
                                   line=dict(color=color, width=5), name=f'a{["1","2","3"][["red","green","blue"].index(color)]}')))
    
    # Reciprocal lattice vectors
    for vec, color in zip([b1,b2,b3], ['magenta','cyan','orange']):
        fig.add_trace(go.Scatter3d(x=[0,vec[0]], y=[0,vec[1]], z=[0,vec[2]],
                                   mode='lines+markers', marker=dict(size=4),
                                   line=dict(color=color, width=5), name=f'b{["1","2","3"][["magenta","cyan","orange"].index(color)]}')))
    
    fig.update_layout(scene=dict(aspectmode='cube'))
    fig.show()

if __name__ == "__main__":
    a1 = np.array([1,0,0])
    a2 = np.array([0,1,0])
    a3 = np.array([0,0,1])
    b1, b2, b3 = reciprocal_lattice(a1,a2,a3)
    plot_lattice(a1,a2,a3,b1,b2,b3)

