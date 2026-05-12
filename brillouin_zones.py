"""
brillouin_zones.py
"""
import os
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot

try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    SHAPELY_OK = True
except ImportError:
    SHAPELY_OK = False

ZONE_COLORS = [
    '#00f0ff', '#00ff9d', '#ffb800', '#ff3e8a',
    '#9b5de5', '#4cc9f0', '#f72585', '#7bed9f',
    '#ffd32a', '#e040fb'
]

ZONE_FILL_RGBA = [
    'rgba(0,240,255,{a})',   'rgba(0,255,157,{a})',  'rgba(255,184,0,{a})',
    'rgba(255,62,138,{a})',  'rgba(155,93,229,{a})', 'rgba(76,201,240,{a})',
    'rgba(247,37,133,{a})',  'rgba(123,237,159,{a})','rgba(255,210,42,{a})',
    'rgba(224,64,251,{a})'
]

DARK_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,15,26,1)',
    font=dict(family='Space Mono, monospace', color='#6a8aaa', size=10),
    margin=dict(l=50, r=20, t=40, b=50),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1a2840',
                borderwidth=1, font=dict(size=9)),
)


def get_lattice_vectors(lattice='square', a=1.0, b=1.0, angle=120):
    if lattice == 'square':
        a1 = np.array([a, 0.0])
        a2 = np.array([0.0, a])
    elif lattice == 'rectangular':
        a1 = np.array([a, 0.0])
        a2 = np.array([0.0, b])
    else:
        th = np.radians(angle)
        a1 = np.array([a, 0.0])
        a2 = np.array([a * np.cos(th), a * np.sin(th)])

    area = a1[0] * a2[1] - a1[1] * a2[0]
    b1   = 2 * np.pi * np.array([ a2[1], -a2[0]]) / area
    b2   = 2 * np.pi * np.array([-a1[1],  a1[0]]) / area
    return a1, a2, b1, b2


def generate_reciprocal_points(b1, b2, N=6):
    pts = []
    for n1 in range(-N, N + 1):
        for n2 in range(-N, N + 1):
            pts.append(n1 * b1 + n2 * b2)
    return np.array(pts)


def _bragg_list(b1, b2, n_zones, max_vecs=96):
    """Reciprocal vectors G (excluding 0), sorted by |G| for stable plane ordering."""
    pts   = generate_reciprocal_points(b1, b2, N=max(n_zones + 4, 8))
    dists = np.linalg.norm(pts, axis=1)
    bragg = pts[dists > 1e-9]
    order = np.argsort(np.sum(bragg * bragg, axis=1))
    bragg = bragg[order]
    return [(float(G[0]), float(G[1]), float(np.dot(G, G))) for G in bragg[:max_vecs]]


def _zone_index_ray(kx, ky, bragg_pre):
    """
    Extended Brillouin zone index: 1 + number of distinct Bragg planes
    k·G = |G|²/2 intersected along the open segment (0, k) from Γ.

    This matches the usual textbook construction (count crossings of perpendicular
    bisectors to reciprocal lattice vectors as one moves outward from Γ).
    """
    kn = float(np.hypot(kx, ky))
    if kn < 1e-14:
        return 1
    khx, khy = kx / kn, ky / kn
    ts = []
    for Gx, Gy, G2 in bragg_pre:
        kg = khx * Gx + khy * Gy
        if kg <= 1e-14:
            continue
        s = 0.5 * G2 / kg
        if s <= 1e-14 or s >= kn * (1.0 - 1e-9):
            continue
        ts.append(s)
    if not ts:
        return 1
    ts.sort()
    nuniq = 1
    prev = ts[0]
    tol = max(1e-12, 1e-7 * kn)
    for s in ts[1:]:
        if abs(s - prev) > tol:
            nuniq += 1
            prev = s
    return 1 + nuniq


def compute_zone_map(b1, b2, n_zones, grid_n=None):
    if grid_n is None:
        grid_n = 240 if os.environ.get('RENDER', '').lower() == 'true' else 350
    b_max  = np.linalg.norm(b1) * (n_zones + 2.5)
    kx_arr = np.linspace(-b_max, b_max, grid_n)
    ky_arr = np.linspace(-b_max, b_max, grid_n)
    KX, KY = np.meshgrid(kx_arr, ky_arr)
    bragg_pre = _bragg_list(b1, b2, n_zones)

    zone_map = np.empty(KX.size, dtype=np.int32)
    i = 0
    for kx, ky in zip(KX.ravel(), KY.ravel()):
        zone_map[i] = _zone_index_ray(kx, ky, bragg_pre)
        i += 1

    zone_map = zone_map.reshape(grid_n, grid_n)
    return kx_arr, ky_arr, zone_map


def build_zone_polygons(kx_arr, ky_arr, zone_map, n_zones, zones_to_show=None):
    if zones_to_show is None:
        zones_to_show = set(range(1, n_zones + 1))
    else:
        zones_to_show = set(zones_to_show)
    if not SHAPELY_OK:
        return [None] * n_zones

    dk        = kx_arr[1] - kx_arr[0]
    zone_polys = []
    for z in range(1, n_zones + 1):
        if z not in zones_to_show:
            zone_polys.append(None)
            continue
        mask      = (zone_map == z)
        iy_idx, ix_idx = np.where(mask)
        if len(ix_idx) == 0:
            zone_polys.append(None)
            continue
        squares = []
        for iy, ix in zip(iy_idx, ix_idx):
            cx, cy = kx_arr[ix], ky_arr[iy]
            squares.append(Polygon([
                (cx - dk/2, cy - dk/2), (cx + dk/2, cy - dk/2),
                (cx + dk/2, cy + dk/2), (cx - dk/2, cy + dk/2)
            ]))
        try:
            poly = unary_union(squares)
            zone_polys.append(poly)
        except Exception:
            zone_polys.append(None)
    return zone_polys


def poly_to_traces(poly, zone_num, alpha=0.22):
    if poly is None or (hasattr(poly, 'is_empty') and poly.is_empty):
        return []

    zix       = max(zone_num - 1, 0)
    color     = ZONE_COLORS[zix % len(ZONE_COLORS)]
    fill_rgba = ZONE_FILL_RGBA[zix % len(ZONE_FILL_RGBA)].format(a=alpha)
    traces    = []

    def add_ring(coords):
        xs, ys = zip(*coords)
        xs = list(xs) + [xs[0]]
        ys = list(ys) + [ys[0]]
        traces.append(go.Scatter(
            x=xs, y=ys, mode='lines',
            fill='toself', fillcolor=fill_rgba,
            line=dict(color=color, width=1.5),
            name=f'Zone {zone_num}',
            showlegend=False,
            hoverinfo='skip'
        ))

    if isinstance(poly, Polygon):
        add_ring(poly.exterior.coords)
    elif isinstance(poly, MultiPolygon):
        for part in poly.geoms:
            add_ring(part.exterior.coords)
    return traces


def plot_brillouin_zones(lattice='square', a=1.0, b=1.0,
                         angle=120, n_zones=4, show_grid=True,
                         zones_to_show=None):
    """
    zones_to_show: iterable of zone indices (1..n_zones) to draw; if None, draw 1..n_zones.
    When multiple zones are selected, fills are drawn on top (semi-transparent overlap).
    """
    n_zones = max(1, min(int(n_zones), 10))
    _, _, b1, b2 = get_lattice_vectors(lattice, a, b, angle)

    if zones_to_show is None:
        z_show = tuple(range(1, n_zones + 1))
    else:
        z_show = tuple(sorted({int(z) for z in zones_to_show if 1 <= int(z) <= n_zones}))
        if not z_show:
            z_show = tuple(range(1, n_zones + 1))

    kx_arr, ky_arr, zone_map = compute_zone_map(b1, b2, n_zones)
    zone_polys = build_zone_polygons(
        kx_arr, ky_arr, zone_map, n_zones, zones_to_show=set(z_show))

    z_label = ','.join(str(z) for z in z_show) if len(z_show) <= 5 else '…'
    layout_kw                  = dict(**DARK_LAYOUT)
    layout_kw['title']         = dict(
        text=f'Brillouin zones {z_label} ({lattice.capitalize()}, max {n_zones})',
        font=dict(size=10, color='#2e4460'), x=0.01)
    layout_kw['xaxis']         = dict(
        title='kx (rad/A)', gridcolor='#1a2840',
        zerolinecolor='rgba(255,255,255,0.13)', linecolor='#1a2840',
        scaleanchor='y', scaleratio=1, tickfont=dict(size=8, color='#6a8aaa'))
    layout_kw['yaxis']         = dict(
        title='ky (rad/A)', gridcolor='#1a2840',
        zerolinecolor='rgba(255,255,255,0.13)', linecolor='#1a2840',
        tickfont=dict(size=8, color='#6a8aaa'))
    layout_kw['showlegend']    = True

    fig = go.Figure(layout=go.Layout(**layout_kw))

    if not SHAPELY_OK:
        # Fallback: scatter plot of zone-coloured pixels
        for z in z_show:
            mask        = (zone_map == z)
            iy_idx, ix_idx = np.where(mask)
            if len(ix_idx) == 0:
                continue
            xs = kx_arr[ix_idx]
            ys = ky_arr[iy_idx]
            col = ZONE_COLORS[(z - 1) % len(ZONE_COLORS)]
            fig.add_trace(go.Scatter(
                x=xs[::4], y=ys[::4], mode='markers',
                marker=dict(color=col, size=2, opacity=0.5),
                name=f'Zone {z}', showlegend=True))
    else:
        for z_idx, poly in enumerate(zone_polys):
            znum = z_idx + 1
            if znum not in z_show:
                continue
            alpha = max(0.09, 0.26 - 0.018 * (len(z_show) - 1))
            for tr in poly_to_traces(poly, znum, alpha=alpha):
                fig.add_trace(tr)
            col = ZONE_COLORS[z_idx % len(ZONE_COLORS)]
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=10, color=col, symbol='square'),
                name=f'Zone {znum}', showlegend=True))

    if show_grid:
        pts = generate_reciprocal_points(b1, b2, N=4)
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode='markers',
            marker=dict(size=4, color='rgba(255,255,255,0.27)', symbol='circle'),
            name='G points', showlegend=True))

    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers+text',
        marker=dict(size=9, color='#ffffff',
                    line=dict(color='#00f0ff', width=2)),
        text=['Gamma'], textposition='top right',
        textfont=dict(color='#00f0ff', size=11),
        name='Gamma point', showlegend=True))

    b_scale = max(np.linalg.norm(b1), np.linalg.norm(b2))
    lim     = b_scale * (n_zones + 1.5)
    fig.update_layout(xaxis_range=[-lim, lim], yaxis_range=[-lim, lim])

    return plot(fig, output_type='div', include_plotlyjs=False)
