"""
brillouin_zones.py
Computes 1st through Nth Brillouin zones for 2D lattices
using the Wigner-Seitz cell construction via perpendicular bisectors.
Supports: square, rectangular, hexagonal lattices.
"""

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import plotly.graph_objs as go
from plotly.offline import plot

# ── Colors per zone ──────────────────────────────────────────
ZONE_COLORS = [
    '#00f0ff', '#00ff9d', '#ffb800', '#ff3e8a', '#9b5de5',
    '#4cc9f0', '#f72585', '#7bed9f', '#ffd32a', '#e040fb'
]
ZONE_FILL = [
    'rgba(0,240,255,{a})', 'rgba(0,255,157,{a})', 'rgba(255,184,0,{a})',
    'rgba(255,62,138,{a})', 'rgba(155,93,229,{a})', 'rgba(76,201,240,{a})',
    'rgba(247,37,133,{a})', 'rgba(123,237,159,{a})', 'rgba(255,210,42,{a})',
    'rgba(224,64,251,{a})'
]

DARK_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,15,26,1)',
    font=dict(family='Space Mono, monospace', color='#6a8aaa', size=10),
    margin=dict(l=50, r=20, t=40, b=50),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1a2840', borderwidth=1,
                font=dict(size=9)),
)


def get_lattice_vectors(lattice='square', a=1.0, b=1.0, angle=120):
    """Return real-space primitive vectors and reciprocal vectors."""
    if lattice == 'square':
        a1 = np.array([a, 0])
        a2 = np.array([0, a])
    elif lattice == 'rectangular':
        a1 = np.array([a, 0])
        a2 = np.array([0, b])
    elif lattice == 'hexagonal':
        th = np.radians(angle)
        a1 = np.array([a, 0])
        a2 = np.array([a * np.cos(th), a * np.sin(th)])
    else:
        a1 = np.array([a, 0])
        a2 = np.array([0, a])

    # 2D reciprocal vectors: b_i · a_j = 2π δ_ij
    area = a1[0] * a2[1] - a1[1] * a2[0]
    b1 = 2 * np.pi * np.array([ a2[1], -a2[0]]) / area
    b2 = 2 * np.pi * np.array([-a1[1],  a1[0]]) / area
    return a1, a2, b1, b2


def generate_reciprocal_points(b1, b2, N=6):
    """Generate grid of reciprocal lattice points around origin."""
    pts = []
    for n1 in range(-N, N+1):
        for n2 in range(-N, N+1):
            pts.append(n1 * b1 + n2 * b2)
    return np.array(pts)


def wigner_seitz_zone(pts, center, clip_radius=None):
    """
    Compute Wigner-Seitz cell around 'center' using perpendicular bisectors
    of vectors to neighboring points. Returns a Shapely Polygon.
    """
    if clip_radius is None:
        clip_radius = np.linalg.norm(pts[np.argsort(np.linalg.norm(pts - center, axis=1))[1]]) * 3

    # Start with large square
    cell = Polygon([
        (-clip_radius, -clip_radius), (clip_radius, -clip_radius),
        (clip_radius, clip_radius), (-clip_radius, clip_radius)
    ])

    for pt in pts:
        diff = pt - center
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            continue
        # Midpoint and normal
        mid  = (center + pt) / 2.0
        norm = diff / dist
        # Half-plane: all points p where (p - mid) · norm <= 0
        # Build a big rectangle on the correct side
        perp = np.array([-norm[1], norm[0]])
        big  = clip_radius * 2
        p1 = mid - big * perp - big * norm
        p2 = mid + big * perp - big * norm
        p3 = mid + big * perp
        p4 = mid - big * perp
        half_plane = Polygon([p1, p2, p3, p4])
        try:
            cell = cell.intersection(half_plane)
        except Exception:
            pass
        if cell.is_empty:
            break

    return cell


def compute_brillouin_zones(lattice='square', a=1.0, b=1.0, angle=120, n_zones=4):
    """
    Returns list of Shapely polygons for zones 1 through n_zones.
    Uses the standard construction: zone N = set of k-points reached by
    exactly N-1 Bragg plane crossings from Gamma.
    """
    _, _, b1, b2 = get_lattice_vectors(lattice, a, b, angle)
    pts = generate_reciprocal_points(b1, b2, N=max(n_zones + 3, 7))
    origin = np.array([0.0, 0.0])

    # Sort points by distance from origin
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts_sorted = pts[order]
    dists_sorted = dists[order]

    # Non-origin points (Bragg planes defined by each)
    nonzero = dists_sorted > 1e-9
    bragg_pts = pts_sorted[nonzero]

    # For each Bragg point, define the bisector half-plane containing origin
    # Zone N is the set of k reached after crossing exactly N-1 planes

    # We compute zone N as: (N-th WS cell of scaled lattice) minus union(zones 1..N-1)
    # More robust: use distance-based zone assignment on a dense k-grid

    # Build dense k-grid
    b_max = np.linalg.norm(b1) * (n_zones + 2)
    grid_n = 400
    kx_arr = np.linspace(-b_max, b_max, grid_n)
    ky_arr = np.linspace(-b_max, b_max, grid_n)
    KX, KY = np.meshgrid(kx_arr, ky_arr)
    k_flat = np.stack([KX.ravel(), KY.ravel()], axis=1)

    # For each k-point, count how many Bragg planes it crosses from origin
    # = number of G-vectors such that |k - G/2| < |G/2|  ↔  k·G > G²/2
    zone_map = np.ones(len(k_flat), dtype=int)  # start at zone 1

    for G in bragg_pts[:min(len(bragg_pts), 60)]:
        G2_half = np.dot(G, G) / 2.0
        # k is beyond bisector if k·G > G²/2
        crossed = (k_flat @ G) > G2_half + 1e-10
        zone_map += crossed.astype(int)

    zone_map = zone_map.reshape(grid_n, grid_n)

    # Build polygons per zone from the grid (contour approach)
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union

    dk = kx_arr[1] - kx_arr[0]
    zone_polys = []
    for z in range(1, n_zones + 1):
        mask = (zone_map == z)
        if not mask.any():
            zone_polys.append(None)
            continue
        # Build squares for each pixel in zone
        squares = []
        iy_idx, ix_idx = np.where(mask)
        for iy, ix in zip(iy_idx, ix_idx):
            cx, cy = kx_arr[ix], ky_arr[iy]
            squares.append(Polygon([
                (cx - dk/2, cy - dk/2), (cx + dk/2, cy - dk/2),
                (cx + dk/2, cy + dk/2), (cx - dk/2, cy + dk/2)
            ]))
        poly = unary_union(squares)
        zone_polys.append(poly)

    return zone_polys, b1, b2, bragg_pts


def poly_to_traces(poly, zone_idx, show_fill=True, alpha=0.25):
    """Convert a Shapely polygon (or MultiPolygon) to Plotly trace list."""
    color = ZONE_COLORS[zone_idx % len(ZONE_COLORS)]
    fill_col = ZONE_FILL[zone_idx % len(ZONE_FILL)].format(a=alpha)
    traces = []

    def add_ring(coords):
        xs, ys = zip(*coords)
        xs = list(xs) + [xs[0]]
        ys = list(ys) + [ys[0]]
        fill = 'toself' if show_fill else 'none'
        traces.append(go.Scatter(
            x=xs, y=ys, mode='lines',
            fill=fill, fillcolor=fill_col,
            line=dict(color=color, width=1.5),
            name=f'Zone {zone_idx + 1}',
            showlegend=False,
            hoverinfo='skip'
        ))

    if poly is None or poly.is_empty:
        return traces
    if isinstance(poly, Polygon):
        add_ring(poly.exterior.coords)
    elif isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            add_ring(p.exterior.coords)
    return traces


def plot_brillouin_zones(lattice='square', a=1.0, b=1.0,
                         angle=120, n_zones=4, show_grid=True):
    """Generate full Plotly figure of BZ 1..n_zones."""
    n_zones = max(1, min(int(n_zones), 10))
    zone_polys, b1, b2, bragg_pts = compute_brillouin_zones(
        lattice, a, b, angle, n_zones)

    layout_kw = dict(**DARK_LAYOUT)
    layout_kw['title'] = dict(
        text=f'Brillouin Zones 1–{n_zones} ({lattice.capitalize()} Lattice)',
        font=dict(size=10, color='#2e4460'), x=0.01)
    layout_kw['xaxis'] = dict(
        title='kₓ (rad/Å)', gridcolor='#1a2840',
        zerolinecolor='#ffffff22', linecolor='#1a2840',
        scaleanchor='y', scaleratio=1, tickfont=dict(size=8))
    layout_kw['yaxis'] = dict(
        title='kᵧ (rad/Å)', gridcolor='#1a2840',
        zerolinecolor='#ffffff22', linecolor='#1a2840', tickfont=dict(size=8))
    layout_kw['showlegend'] = True
    layout_kw['legend'] = dict(
        bgcolor='rgba(0,0,0,0)', bordercolor='#1a2840',
        borderwidth=1, font=dict(size=9))
    fig = go.Figure(layout=go.Layout(**layout_kw))

    # Add zone fills
    for z_idx, poly in enumerate(zone_polys):
        if poly is None:
            continue
        for tr in poly_to_traces(poly, z_idx, show_fill=True, alpha=0.22):
            fig.add_trace(tr)
        # Add legend entry
        color = ZONE_COLORS[z_idx % len(ZONE_COLORS)]
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=color, symbol='square'),
            name=f'Zone {z_idx + 1}', showlegend=True))

    # Reciprocal lattice points
    if show_grid:
        _, _, b1g, b2g = get_lattice_vectors(lattice, a, b, angle)
        pts = generate_reciprocal_points(b1g, b2g, N=4)
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode='markers',
            marker=dict(size=4, color='#ffffff44', symbol='circle'),
            name='G points', showlegend=True))

    # Gamma point
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers+text',
        marker=dict(size=8, color='#ffffff', symbol='circle',
                    line=dict(color='#00f0ff', width=2)),
        text=['Γ'], textposition='top right',
        textfont=dict(color='#00f0ff', size=11),
        name='Γ point', showlegend=True))

    b_scale = max(np.linalg.norm(b1), np.linalg.norm(b2))
    lim = b_scale * (n_zones + 1.5)
    fig.update_layout(
        xaxis_range=[-lim, lim],
        yaxis_range=[-lim, lim],
    )

    return plot(fig, output_type='div', include_plotlyjs=False)
