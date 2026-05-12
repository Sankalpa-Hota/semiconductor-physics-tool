from flask import Flask, render_template, request, jsonify
from chatbot import chatbot_bp
from brillouin_zones import plot_brillouin_zones
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import os
import json
import urllib.request
import urllib.parse

import fermi_boltzmann as fb
import drude_model as dm
import phonon_scattering as ps
import kronig_penney as kp
import reciprocal_lattice as rl
import oq_sections as oqs

app = Flask(__name__)
app.register_blueprint(chatbot_bp)

# Physical constants
k_B  = 1.380649e-23
hbar = 1.0545718e-34
q    = 1.60218e-19
m0   = 9.10938e-31
eV   = 1.60218e-19

COLORS = [
    '#00f0ff', '#00ff9d', '#ffb800', '#ff3e8a',
    '#9b5de5', '#4cc9f0', '#7bed9f', '#ffd32a', '#f72585'
]

DARK_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,15,26,1)',
    font=dict(family='Space Mono, monospace', color='#6a8aaa', size=10),
    xaxis=dict(gridcolor='#1a2840', zerolinecolor='#1a2840',
               linecolor='#1a2840', tickfont=dict(size=9)),
    yaxis=dict(gridcolor='#1a2840', zerolinecolor='#1a2840',
               linecolor='#1a2840', tickfont=dict(size=9)),
    margin=dict(l=55, r=18, t=38, b=50),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1a2840',
                borderwidth=1, font=dict(size=9)),
)


def make_layout(title='', **kw):
    d = dict(**DARK_LAYOUT)
    d['title'] = dict(text=title, font=dict(size=10, color='#2e4460'), x=0.01)
    d.update(kw)
    return go.Layout(**d)


def gf(form, name, default):
    try:
        v = form.get(name, '').strip()
        return float(v) if v else default
    except Exception:
        return default


def gi(form, name, default, lo=None, hi=None):
    """Bounded integer from a form field."""
    try:
        raw = form.get(name, '')
        v = int(round(float(raw.strip() or default)))
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        return v
    except (TypeError, ValueError):
        return default


BZ_LATTICES = frozenset({'square', 'rectangular', 'hexagonal'})

# Plots are built only when the user checks them (saves CPU/RAM on small hosts).
PLOT_KEYS = frozenset({
    'fd', 'dos', 'ni_T', 'mu_T', 'carr_T', 'cond_T',
    'ef_dop', 'band', 'rho_dop', 'eg_T',
    'iv', 'schot', 'hall', 'depl',
    'kp1d', 'kp2d', 'kp3d',
    'bz_plot', 'phonon', 'recip',
    'ek_comp', 'pn_x', 'dos_qw', 'tauc', 'sdh',
})

PLOT_PLACEHOLDER = (
    "<div class=\"plot-placeholder\">"
    "<p><strong>Plot not loaded.</strong> Tick it above and click "
    "<strong>Apply parameters &amp; load selected plots</strong>, or press "
    "<strong>Render this</strong> on the card.</p>"
    "<p class=\"plot-ph-hint\">Guided mode only builds figures you ask for.</p>"
    "</div>"
)

def parse_open_plots(form):
    """Plot keys from checkbox list (plot_pick) ∪ hidden open_plots; guided mode if both empty."""
    keys = set()
    for item in form.getlist('plot_pick'):
        k = (item or '').strip()
        if k in PLOT_KEYS:
            keys.add(k)
    raw = form.get('open_plots')
    if raw:
        for k in raw.split(','):
            k = k.strip()
            if k in PLOT_KEYS:
                keys.add(k)
    return frozenset(keys)


def fetch_wikipedia_extract(title, timeout=2.5, max_len=480):
    """Short extract from Wikipedia REST API (best-effort; offline-safe)."""
    if not title:
        return ''
    t = title.replace(' ', '_')
    url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(t, safe="")}'
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'OPEN-Quantum/1.0 (education; +https://github.com/)'},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode('utf-8'))
        ext = (data.get('extract') or '').strip()
        if not ext:
            return ''
        return (ext[:max_len] + '…') if len(ext) > max_len else ext
    except Exception:
        return ''


PLOT_TITLES = {
    'fd': 'Fermi–Dirac vs energy',
    'dos': '3D DOS and occupation',
    'dos_qw': 'Quantum-well 2D DOS (stairs)',
    'ni_T': 'Intrinsic nᵢ vs T',
    'mu_T': 'Mobility vs T',
    'carr_T': 'n, p vs T',
    'cond_T': 'Conductivity vs T',
    'sdh': 'Shubnikov–de Haas (model)',
    'ef_dop': 'Fermi level vs doping',
    'band': 'Flat-band diagram',
    'rho_dop': 'Resistivity vs doping',
    'eg_T': 'Bandgap vs T (Varshni)',
    'tauc': 'Tauc plot',
    'iv': 'p–n junction I–V',
    'schot': 'Schottky bands',
    'hall': 'Hall voltage vs B',
    'depl': 'Depletion width',
    'pn_x': 'p–n bands (x)',
    'ek_comp': 'Direct vs indirect E(k)',
    'kp1d': 'Kronig–Penney E(k)',
    'kp2d': 'KP vs barrier height',
    'kp3d': 'KP 3D surface',
    'bz_plot': 'Brillouin zones (zone index 1–10)',
    'phonon': 'Phonon dispersion',
    'recip': 'Real vs reciprocal lattice (3D)',
}


def parse_recip_form(form):
    """Vertices R0… in Cartesian coords; first vertex is the cell corner."""
    n = gi(form, 'recip_n', 4, lo=3, hi=8)
    dv = rl.DEFAULT_VERTICES
    rows = []
    for i in range(n):
        if i < len(dv):
            dx, dy, dz = float(dv[i, 0]), float(dv[i, 1]), float(dv[i, 2])
        else:
            dx, dy, dz = 0.0, float(i) * 0.2, 0.0
        rows.append([
            gf(form, f'recip_{i}_x', dx),
            gf(form, f'recip_{i}_y', dy),
            gf(form, f'recip_{i}_z', dz),
        ])
    c_axis = gf(form, 'recip_c_axis', 0.6)
    return np.array(rows, dtype=float), n, c_axis


# Hole DOS effective mass (m0); Si ~0.49-0.56 depending on model; keeps VB DOS
# separate from conduction mass tied to the m* slider.
M_P_DOS_RATIO = 0.56


def parse_bz_layers_raw(raw, n_zones_cap=10):
    """Comma-separated zone indices 1..10; 'all' or empty → 1..n_zones_cap (cap ∈ [1,10])."""
    raw = (raw or '').strip()
    n_cap = max(1, min(10, int(n_zones_cap)))
    if not raw or raw.lower() == 'all':
        return tuple(range(1, n_cap + 1))
    out = []
    for p in raw.split(','):
        p = p.strip()
        if not p:
            continue
        try:
            z = int(round(float(p)))
            if 1 <= z <= 10:
                out.append(z)
        except (TypeError, ValueError):
            pass
    return tuple(sorted(set(out))) or tuple(range(1, n_cap + 1))


def parse_bz_layers_json(val, n_zones_cap=10):
    """Accept list, 'all', or comma string from JSON API (indices 1..10)."""
    n_cap = max(1, min(10, int(n_zones_cap)))
    if val is None or val == 'all':
        return tuple(range(1, n_cap + 1))
    if isinstance(val, list):
        out = []
        for x in val:
            try:
                z = int(round(float(x)))
                if 1 <= z <= 10:
                    out.append(z)
            except (TypeError, ValueError):
                pass
        return tuple(sorted(set(out))) or tuple(range(1, n_cap + 1))
    if isinstance(val, str):
        return parse_bz_layers_raw(val, n_cap)
    return tuple(range(1, n_cap + 1))


def simulation_context(
    Nc, Nv, Eg, Nd, T, tau, vF, m_eff_ratio, a, V0_m, b_m,
    bz_lattice, bz_a, bz_b, bz_angle, bz_zones, bz_layers=None,
    recip_vertices=None, recip_c_axis=0.6,
):
    Ec = 0.0
    Ev = -Eg
    V0_J = V0_m * eV
    m_eff = m_eff_ratio * m0
    if bz_layers is None:
        bz_layers = (1, 2, 3, 4)
    if recip_vertices is None:
        recip_vertices = rl.DEFAULT_VERTICES.copy()
    ni = fb.intrinsic_carrier_concentration(Nc, Nv, Eg, T)
    Ef = fb.fermi_level_n_type(Ec, Ev, Nc, Nv, Nd, ni, T)
    n, p = fb.carrier_concentration(Ef, Ec, Ev, Nc, Nv, T)
    mu = dm.mobility_drude(tau, m_eff) * 1e4
    sigma = dm.conductivity(n, mu)
    l_mfp = dm.mean_free_path(vF, tau)
    return {
        'Nc': Nc, 'Nv': Nv, 'Eg': Eg, 'Nd': Nd, 'T': T,
        'tau': tau, 'vF': vF, 'm_eff_ratio': m_eff_ratio,
        'a': a, 'V0': V0_m, 'b': b_m, 'V0_J': V0_J,
        'bz_lattice': bz_lattice, 'bz_a': bz_a, 'bz_b': bz_b,
        'bz_angle': bz_angle, 'bz_zones': bz_zones, 'bz_layers': bz_layers,
        'm_p_dos_ratio': M_P_DOS_RATIO,
        'Ec': Ec, 'Ev': Ev, 'ni': ni, 'Ef': Ef, 'n': n, 'p': p,
        'mu': mu, 'sigma': sigma, 'l_mfp': l_mfp,
        'recip_vertices': np.asarray(recip_vertices, dtype=float),
        'recip_c_axis': float(recip_c_axis),
    }


def render_plot_html(plot_key, ctx):
    """Return Plotly div (or error HTML) for a single key."""
    k = plot_key
    if k == 'fd':
        return plt_fermi_dirac(
            ctx['Nc'], ctx['Nv'], ctx['Eg'], ctx['Nd'], ctx['T'])
    if k == 'dos':
        return plt_dos(
            ctx['m_eff_ratio'], ctx['m_p_dos_ratio'],
            ctx['Ec'], ctx['Ev'], ctx['T'], ctx['Ef'])
    if k == 'ni_T':
        return plt_ni_vs_T(ctx['Nc'], ctx['Nv'], ctx['Eg'])
    if k == 'mu_T':
        return plt_mobility_vs_T(ctx['Nd'])
    if k == 'carr_T':
        return plt_carrier_vs_T(ctx['Nc'], ctx['Nv'], ctx['Eg'], ctx['Nd'])
    if k == 'cond_T':
        return plt_conductivity_vs_T(ctx['Nc'], ctx['Nv'], ctx['Eg'], ctx['Nd'])
    if k == 'ef_dop':
        return plt_ef_vs_doping(ctx['Nc'], ctx['Nv'], ctx['Eg'], ctx['T'])
    if k == 'band':
        return plt_band_diagram(
            ctx['Ef'], ctx['Ec'], ctx['Ev'], ctx['Eg'])
    if k == 'rho_dop':
        return plt_resistivity_vs_doping()
    if k == 'eg_T':
        return plt_bandgap_vs_T(ctx['Eg'])
    if k == 'kp1d':
        return plt_kp_1d(ctx['a'], ctx['V0_J'], ctx['b'])
    if k == 'kp2d':
        return plt_kp_2d(ctx['a'], ctx['V0_J'], ctx['b'])
    if k == 'kp3d':
        return plt_kp_3d(ctx['a'], ctx['V0_J'], ctx['b'])
    if k == 'recip':
        return plt_reciprocal_lattice(
            ctx['recip_vertices'], ctx['recip_c_axis'])
    if k == 'hall':
        return plt_hall(ctx['Nc'], ctx['Nv'], ctx['Eg'], ctx['Nd'])
    if k == 'iv':
        return plt_iv_diode(ctx['T'], ctx['Eg'], ctx['Nc'], ctx['Nv'])
    if k == 'schot':
        return plt_schottky(ctx['Eg'])
    if k == 'phonon':
        return plt_phonon()
    if k == 'depl':
        return plt_depletion(ctx['Nc'], ctx['Nd'], ctx['Eg'])
    if k == 'ek_comp':
        return plt_ek_compare(ctx['Eg'])
    if k == 'pn_x':
        return plt_pn_junction_bands(ctx['Eg'])
    if k == 'dos_qw':
        return plt_dos_qw_2d(ctx['m_eff_ratio'], L_nm=12.0)
    if k == 'tauc':
        return plt_tauc(ctx['Eg'])
    if k == 'sdh':
        return plt_sdh()
    if k == 'bz_plot':
        return plot_brillouin_zones(
            lattice=ctx['bz_lattice'], a=ctx['bz_a'], b=ctx['bz_b'],
            angle=ctx['bz_angle'], n_zones=ctx['bz_zones'],
            zones_to_show=ctx['bz_layers'])
    return PLOT_PLACEHOLDER


def _float_param(data, key, default):
    if key not in data or data[key] is None or data[key] == '':
        return float(default)
    return float(data[key])


def parse_recip_api(data):
    cax = 0.6
    try:
        if data.get('recip_c_axis') not in (None, ''):
            cax = float(data['recip_c_axis'])
    except (TypeError, ValueError):
        cax = 0.6
    rv = data.get('recip_vertices')
    if isinstance(rv, list) and len(rv) >= 3:
        rows = []
        for row in rv[:8]:
            if isinstance(row, (list, tuple)) and len(row) >= 3:
                rows.append([float(row[0]), float(row[1]), float(row[2])])
        if len(rows) >= 3:
            return np.array(rows, dtype=float), cax
    return rl.DEFAULT_VERTICES.copy(), cax


def context_from_api_payload(data):
    """Build simulation_context from JSON (same keys as the main form)."""
    Nc = fb.NC_SI_CM3_300K
    Nv = fb.NV_SI_CM3_300K
    Eg = fb.EG_SI_EV_300K
    Nd = 1e17
    T = 300.0
    tau = 0.24e-15
    vF = 1e6
    m_eff_ratio = 0.26
    a = 5e-10
    V0_m = 10.0
    b_m = 2e-10
    bz_lattice = 'square'
    bz_a, bz_b, bz_angle, bz_zones = 1.0, 1.5, 120.0, 10

    Nc = _float_param(data, 'Nc', Nc)
    Nv = _float_param(data, 'Nv', Nv)
    Eg = _float_param(data, 'Eg', Eg)
    Nd = _float_param(data, 'Nd', Nd)
    T = _float_param(data, 'T', T)
    tau = _float_param(data, 'tau', tau)
    vF = _float_param(data, 'vF', vF)
    m_eff_ratio = _float_param(data, 'm_eff', m_eff_ratio)
    a = _float_param(data, 'a', a)
    V0_m = _float_param(data, 'V0', V0_m)
    b_m = _float_param(data, 'b', b_m)
    bz_a = _float_param(data, 'bz_a', bz_a)
    bz_b = _float_param(data, 'bz_b', bz_b)
    bz_angle = _float_param(data, 'bz_angle', bz_angle)
    bz_zones = 10
    lat = data.get('bz_lattice', bz_lattice)
    if isinstance(lat, str) and lat in BZ_LATTICES:
        bz_lattice = lat

    layer_src = data.get('bz_layers')
    if layer_src is None:
        layer_src = data.get('layers')
    bz_layers = parse_bz_layers_json(layer_src, bz_zones)

    recip_vertices, recip_c_axis = parse_recip_api(data)

    return simulation_context(
        Nc, Nv, Eg, Nd, T, tau, vF, m_eff_ratio, a, V0_m, b_m,
        bz_lattice, bz_a, bz_b, bz_angle, bz_zones, bz_layers=bz_layers,
        recip_vertices=recip_vertices, recip_c_axis=recip_c_axis,
    )


def _n(num):
    """Fewer sample points on Render — speeds up the very heavy / page."""
    n = int(num)
    if os.environ.get('RENDER', '').lower() != 'true':
        return n
    if n <= 24:
        return max(8, int(n * 0.85) or 8)
    return max(24, min(384, int(n * 0.62)))


def pplot(fig):
    return plot(fig, output_type='div', include_plotlyjs=False)


# Physics helpers

def fermi_dirac(E, Ef, T):
    x = (E - Ef) / (k_B * max(T, 1) / eV)
    return 1.0 / (np.exp(np.clip(x, -500, 500)) + 1)


def mu_calc(T, Nd):
    T    = max(float(T), 1.0)
    mu_L = 1350.0 * (T / 300.0) ** -2.3
    if Nd > 0:
        Nd    = max(float(Nd), 1.0)
        denom = Nd * max(np.log(1 + 4.6e13 * T**2 / Nd), 1e-12)
        mu_I  = 4e21 * T**1.5 / denom
    else:
        mu_I = 1e8
    return 1.0 / (1.0/mu_L + 1.0/mu_I)


def sigma_calc(n, mu):
    # n in cm^-3, mu in cm^2/V.s, returns S/m
    return n * 1e6 * mu * 1e-4 * q


def kp_dispersion(a, V0_J, b, N=None):
    if N is None:
        N = _n(600)
    E_arr  = np.linspace(1e-5 * eV, V0_J * 3.5, N)
    k_list = []
    E_list = []
    for E in E_arr:
        try:
            alpha = np.sqrt(2 * m0 * E) / hbar
            if E < V0_J:
                beta = np.sqrt(2 * m0 * (V0_J - E)) / hbar
                term = (np.cos(alpha*(a-b)) * np.cosh(np.clip(beta*b, -300, 300))
                        + (beta**2 - alpha**2) / (2*alpha*beta)
                        * np.sin(alpha*(a-b)) * np.sinh(np.clip(beta*b, -300, 300)))
            else:
                gamma = np.sqrt(2 * m0 * (E - V0_J)) / hbar
                term  = (np.cos(alpha*(a-b)) * np.cos(gamma*b)
                         - (gamma**2 + alpha**2) / (2*alpha*gamma)
                         * np.sin(alpha*(a-b)) * np.sin(gamma*b))
            if -1.0 <= term <= 1.0:
                k_list.append(np.arccos(np.clip(term, -1, 1)) / a)
                E_list.append(E / eV)
        except Exception:
            pass
    return np.array(k_list), np.array(E_list)


# Plot functions

def plt_fermi_dirac(Nc, Nv, Eg, Nd, T_ref):
    """f(E) vs E with Ef(T) recomputed for each curve at fixed Nd (non-degenerate n-type)."""
    Ec, Ev = 0.0, -Eg
    E = np.linspace(Ev - 0.08, Ec + 0.08, _n(600))
    fig = go.Figure(layout=make_layout('Fermi–Dirac (Ef varies with T at fixed Nd)'))
    temps = [100, 200, T_ref, 500, 800]
    for i, Ti in enumerate(temps):
        ni_i = fb.intrinsic_carrier_concentration(Nc, Nv, Eg, Ti)
        Ef_i = fb.fermi_level_n_type(Ec, Ev, Nc, Nv, Nd, ni_i, Ti)
        lw = 2.6 if int(round(Ti)) == int(round(T_ref)) else 2.0
        fig.add_trace(go.Scatter(
            x=fermi_dirac(E, Ef_i, Ti), y=E,
            mode='lines', name=f'{int(Ti)} K, Ef={Ef_i:.3f} eV',
            line=dict(color=COLORS[i % len(COLORS)], width=lw)))
    ni0 = fb.intrinsic_carrier_concentration(Nc, Nv, Eg, T_ref)
    Ef0 = fb.fermi_level_n_type(Ec, Ev, Nc, Nv, Nd, ni0, T_ref)
    fig.add_hline(y=Ef0,
                  line=dict(color='rgba(255,184,0,0.45)', dash='dot', width=1),
                  annotation_text=f'Ef @ {int(T_ref)} K',
                  annotation_font=dict(size=8, color='#ffb800'))
    fig.add_hline(y=Ec, line=dict(color='rgba(0,240,255,0.25)', width=1, dash='dot'))
    fig.add_hline(y=Ev, line=dict(color='rgba(255,62,138,0.25)', width=1, dash='dot'))
    fig.update_layout(xaxis_title='f(E)', yaxis_title='Energy (eV)')
    return pplot(fig)


def plt_dos(m_n_ratio, m_p_ratio, Ec, Ev, T, Ef):
    """
    3D parabolic bands: g(E) ∝ √(E−Ec) (CB), √(Ev−E) (VB), per eV per cm³.
    m_n* from slider; m_p* separate (Si-like valence DOS mass).
    """
    m_cb = max(float(m_n_ratio), 1e-4) * m0
    m_vb = max(float(m_p_ratio), 1e-4) * m0
    E    = np.linspace(Ev - 0.15, Ec + 0.35, _n(800))
    dos_c = np.zeros_like(E)
    dos_v = np.zeros_like(E)
    cb = E > Ec
    vb = E < Ev
    if cb.any():
        dE_J = np.maximum((E[cb] - Ec) * eV, 0.0)
        dos_c[cb] = ((1 / (2 * np.pi ** 2)) * (2 * m_cb / hbar ** 2) ** 1.5
                     * np.sqrt(dE_J) * eV / 1e6)
    if vb.any():
        dE_J = np.maximum((Ev - E[vb]) * eV, 0.0)
        dos_v[vb] = ((1 / (2 * np.pi ** 2)) * (2 * m_vb / hbar ** 2) ** 1.5
                     * np.sqrt(dE_J) * eV / 1e6)
    norm = max(dos_c.max(), dos_v.max(), 1e-30)
    f    = fermi_dirac(E, Ef, T)
    fig  = go.Figure(layout=make_layout('DOS (3D parabolic) & Fermi occupation'))
    fig.add_trace(go.Scatter(x=E, y=dos_c / norm, mode='lines', name='g_c (CB)',
        line=dict(color=COLORS[0], width=2),
        fill='tozeroy', fillcolor='rgba(0,240,255,0.05)'))
    fig.add_trace(go.Scatter(x=E, y=dos_v / norm, mode='lines', name='g_v (VB)',
        line=dict(color=COLORS[3], width=2),
        fill='tozeroy', fillcolor='rgba(255,62,138,0.05)'))
    fig.add_trace(go.Scatter(x=E, y=f, mode='lines', name='f(E)',
        line=dict(color=COLORS[2], width=1.8, dash='dash')))
    fig.add_trace(go.Scatter(x=E, y=(dos_c / norm) * f, mode='lines',
        name='g_c·f (occupied CB)',
        line=dict(color=COLORS[1], width=2),
        fill='tozeroy', fillcolor='rgba(0,255,157,0.05)'))
    fig.add_vline(x=Ef, line=dict(color='rgba(255,184,0,0.35)', dash='dot', width=1))
    fig.update_layout(xaxis_title='Energy (eV)', yaxis_title='Normalized')
    return pplot(fig)


def plt_ni_vs_T(Nc, Nv, Eg):
    T  = np.linspace(150, 900, _n(400))
    ni = np.array([fb.intrinsic_carrier_concentration(Nc, Nv, Eg, Ti) for Ti in T])
    fig = go.Figure(layout=make_layout('ni vs Temperature'))
    fig.add_trace(go.Scatter(x=T, y=ni, mode='lines',
        line=dict(color=COLORS[0], width=2.5),
        fill='tozeroy', fillcolor='rgba(0,240,255,0.04)'))
    fig.update_layout(xaxis_title='T (K)',
                      yaxis_title='ni (cm^-3)', yaxis_type='log')
    return pplot(fig)


def plt_mobility_vs_T(Nd):
    Nd   = max(float(Nd), 1e-30)
    T    = np.linspace(80, 700, _n(400))
    mu   = np.array([mu_calc(Ti, Nd) for Ti in T])
    mu   = np.clip(mu, 1e-6, None)
    mu_L = 1350.0 * (T / 300.0) ** -2.3
    fig  = go.Figure(layout=make_layout('Mobility vs Temperature'))
    fig.add_trace(go.Scatter(x=T, y=mu, mode='lines', name='Total μ',
        line=dict(color=COLORS[1], width=2.5)))
    fig.add_trace(go.Scatter(x=T, y=mu_L, mode='lines', name='Lattice μ',
        line=dict(color=COLORS[0], width=1.5, dash='dash')))
    fig.update_layout(xaxis_title='T (K)', yaxis_title='μ (cm²/V·s)',
                      yaxis_type='linear')
    return pplot(fig)


def plt_carrier_vs_T(Nc, Nv, Eg, Nd):
    T    = np.linspace(150, 700, _n(400))
    Ec   = 0.0
    Ev   = -Eg
    n_a, p_a, ni_a = [], [], []
    for Ti in T:
        ni_v = fb.intrinsic_carrier_concentration(Nc, Nv, Eg, Ti)
        Ef_v = fb.fermi_level_n_type(Ec, Ev, Nc, Nv, Nd, ni_v, Ti)
        n_v, p_v = fb.carrier_concentration(Ef_v, Ec, Ev, Nc, Nv, Ti)
        n_a.append(n_v)
        p_a.append(p_v)
        ni_a.append(ni_v)
    fig = go.Figure(layout=make_layout('Carrier Concentrations vs T'))
    fig.add_trace(go.Scatter(x=T, y=n_a, mode='lines', name='n (electrons)',
        line=dict(color=COLORS[0], width=2)))
    fig.add_trace(go.Scatter(x=T, y=p_a, mode='lines', name='p (holes)',
        line=dict(color=COLORS[3], width=2)))
    fig.add_trace(go.Scatter(x=T, y=ni_a, mode='lines', name='ni',
        line=dict(color=COLORS[1], width=1.5, dash='dash')))
    fig.update_layout(xaxis_title='T (K)',
                      yaxis_title='(cm^-3)', yaxis_type='log')
    return pplot(fig)


def plt_conductivity_vs_T(Nc, Nv, Eg, Nd):
    T    = np.linspace(150, 700, _n(400))
    vals = []
    for Ti in T:
        ni_v = fb.intrinsic_carrier_concentration(Nc, Nv, Eg, Ti)
        n_v  = max(Nd, ni_v)
        mu_v = mu_calc(Ti, Nd)
        vals.append(sigma_calc(n_v, mu_v))
    fig = go.Figure(layout=make_layout('Conductivity vs Temperature'))
    fig.add_trace(go.Scatter(x=T, y=vals, mode='lines',
        line=dict(color=COLORS[3], width=2.5),
        fill='tozeroy', fillcolor='rgba(255,62,138,0.04)'))
    fig.update_layout(xaxis_title='T (K)',
                      yaxis_title='sigma (S/m)', yaxis_type='log')
    return pplot(fig)


def plt_ef_vs_doping(Nc, Nv, Eg, T):
    Nd_arr = np.logspace(13, 20, _n(300))
    Ec     = 0.0
    Ev     = -Eg
    Ef_arr = [
        fb.fermi_level_n_type(
            Ec, Ev, Nc, Nv, Nd_i,
            fb.intrinsic_carrier_concentration(Nc, Nv, Eg, T), T)
        for Nd_i in Nd_arr]
    fig    = go.Figure(layout=make_layout('Fermi Level vs Doping (n-type)'))
    fig.add_trace(go.Scatter(x=Nd_arr, y=Ef_arr, mode='lines',
        line=dict(color=COLORS[4], width=2.5)))
    fig.add_hline(y=Ec,
                  line=dict(color='rgba(0,240,255,0.3)', dash='dot'),
                  annotation_text='Ec',
                  annotation_font=dict(color='#00f0ff', size=9))
    fig.add_hline(y=Ev,
                  line=dict(color='rgba(255,62,138,0.3)', dash='dot'),
                  annotation_text='Ev',
                  annotation_font=dict(color='#ff3e8a', size=9))
    fig.update_layout(xaxis_title='Nd (cm^-3)', xaxis_type='log',
                      yaxis_title='Ef (eV)')
    return pplot(fig)


def plt_band_diagram(Ef, Ec, Ev, Eg):
    x   = [0, 2]
    fig = go.Figure(layout=make_layout('Energy Band Diagram'))
    fig.add_trace(go.Scatter(x=x, y=[Ec, Ec], mode='lines', name='Ec (CB)',
        line=dict(color=COLORS[0], width=3)))
    fig.add_trace(go.Scatter(x=x, y=[Ev, Ev], mode='lines', name='Ev (VB)',
        line=dict(color=COLORS[3], width=3)))
    fig.add_trace(go.Scatter(x=x, y=[Ef, Ef], mode='lines', name='Ef',
        line=dict(color=COLORS[2], width=2, dash='dash')))
    mid = (Ec + Ev) / 2.0
    fig.add_trace(go.Scatter(x=x, y=[mid, mid], mode='lines', name='Ei',
        line=dict(color='rgba(255,255,255,0.13)', width=1, dash='dot')))
    fig.add_hrect(y0=Ev, y1=Ec,
                  fillcolor='rgba(0,240,255,0.03)', line_width=0)
    fig.update_layout(xaxis=dict(visible=False), yaxis_title='Energy (eV)')
    return pplot(fig)


def plt_resistivity_vs_doping():
    Nd_arr = np.logspace(13, 20, _n(300))
    rho_n, rho_p = [], []
    for Nd in Nd_arr:
        mu_n  = mu_calc(300, Nd)
        mu_p  = mu_n * 0.47
        sig_n = sigma_calc(Nd, mu_n)
        sig_p = sigma_calc(Nd, mu_p)
        rho_n.append(1.0/sig_n if sig_n > 0 else 1e10)
        rho_p.append(1.0/sig_p if sig_p > 0 else 1e10)
    fig = go.Figure(layout=make_layout('Resistivity vs Doping'))
    fig.add_trace(go.Scatter(x=Nd_arr, y=rho_n, mode='lines', name='n-type',
        line=dict(color=COLORS[0], width=2)))
    fig.add_trace(go.Scatter(x=Nd_arr, y=rho_p, mode='lines', name='p-type',
        line=dict(color=COLORS[3], width=2)))
    fig.update_layout(xaxis_title='Doping (cm^-3)', xaxis_type='log',
                      yaxis_title='rho (ohm.m)', yaxis_type='log')
    return pplot(fig)


def plt_bandgap_vs_T(Eg_300=1.12):
    T     = np.linspace(1, 600, _n(400))
    alpha = 4.73e-4
    beta  = 636.0
    ref   = Eg_300 + alpha * 300**2 / (300 + beta)
    Eg    = ref - alpha * T**2 / (T + beta)
    fig   = go.Figure(layout=make_layout('Bandgap vs Temperature (Varshni)'))
    fig.add_trace(go.Scatter(x=T, y=Eg, mode='lines',
        line=dict(color=COLORS[2], width=2.5),
        fill='tozeroy', fillcolor='rgba(255,184,0,0.04)'))
    fig.update_layout(xaxis_title='T (K)', yaxis_title='Eg (eV)')
    return pplot(fig)


def plt_kp_1d(a, V0_J, b):
    k_vals, E_vals = kp_dispersion(a, V0_J, b)
    if len(k_vals) == 0:
        return ("<p style='color:#6a8aaa;padding:1rem;"
                "font-family:monospace;font-size:0.75rem'>"
                "No allowed bands — adjust V0 or barrier width.</p>")
    fig = go.Figure(layout=make_layout('Kronig-Penney Band Structure E(k)'))
    fig.add_trace(go.Scatter(
        x=k_vals/1e10, y=E_vals, mode='markers',
        marker=dict(color=COLORS[0], size=2.5, opacity=0.8),
        name='Allowed E'))
    fig.update_layout(xaxis_title='k (1/A)', yaxis_title='E (eV)')
    return pplot(fig)


def plt_kp_2d(a, V0_J, b):
    V0_range      = np.linspace(0.5*eV, V0_J * 2.5, _n(30))
    all_k, all_E, all_V = [], [], []
    for V in V0_range:
        k, E = kp_dispersion(a, V, b, N=_n(200))
        if len(k):
            all_k.extend(k)
            all_E.extend(E)
            all_V.extend([V/eV] * len(k))
    if not all_k:
        return ("<p style='color:#6a8aaa;padding:1rem;"
                "font-family:monospace;font-size:0.75rem'>"
                "Insufficient data for this parameter range.</p>")
    fig = go.Figure(layout=make_layout('KP Band Map vs Barrier Height'))
    fig.add_trace(go.Scatter(
        x=np.array(all_k)/1e10, y=all_E, mode='markers',
        marker=dict(
            color=all_V, colorscale='Plasma', size=2, opacity=0.65,
            colorbar=dict(title='V0 (eV)', tickfont=dict(size=8))),
        name='Allowed E'))
    fig.update_layout(xaxis_title='k (1/A)', yaxis_title='E (eV)')
    return pplot(fig)


def plt_kp_3d(a, V0_J, b):
    k_vals, E_vals = kp_dispersion(a, V0_J, b, N=_n(200))
    if len(k_vals) < 4:
        return ("<p style='color:#6a8aaa;padding:1rem;"
                "font-family:monospace;font-size:0.75rem'>"
                "Insufficient KP data for 3D surface plot.</p>")
    k_all = np.concatenate([-k_vals[::-1], k_vals])
    E_all = np.concatenate([E_vals[::-1],  E_vals])
    k_axis = k_all / 1e10
    kx, ky = np.meshgrid(k_axis, k_axis)
    e0, e1 = E_all[0], E_all[-1]
    try:
        ex = np.interp(np.abs(kx), k_axis, E_all, left=e0, right=e1)
        ey = np.interp(np.abs(ky), k_axis, E_all, left=e0, right=e1)
        Ez = np.sqrt(ex ** 2 * 0.5 + ey ** 2 * 0.5)
    except Exception:
        r = np.hypot(kx, ky)
        Ez = np.interp(r, k_axis, E_all, left=e0, right=e1)

    fig = go.Figure(layout=make_layout('3D Band Structure E(kx, ky)'))
    fig.add_trace(go.Surface(
        x=kx, y=ky, z=Ez,
        colorscale='Plasma', opacity=0.88,
        colorbar=dict(title='E (eV)',
                      tickfont=dict(size=8, color='#6a8aaa')),
        contours=dict(z=dict(show=True,
                             color='rgba(0,240,255,0.3)', width=1))
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='kx (1/A)', gridcolor='#1a2840',
                       backgroundcolor='rgba(10,15,26,1)'),
            yaxis=dict(title='ky (1/A)', gridcolor='#1a2840',
                       backgroundcolor='rgba(10,15,26,1)'),
            zaxis=dict(title='E (eV)',   gridcolor='#1a2840',
                       backgroundcolor='rgba(10,15,26,1)'),
            bgcolor='rgba(10,15,26,1)',
        ),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
    )
    return pplot(fig)


def plt_reciprocal_lattice(vertices, c_axis=0.6):
    return rl.plot_reciprocal_lattice_3d(vertices=vertices, c_axis=c_axis)


def plt_hall(Nc, Nv, Eg, Nd):
    T   = 300
    ni  = fb.intrinsic_carrier_concentration(Nc, Nv, Eg, T)
    n   = max(Nd, ni)
    B   = np.linspace(0, 3, _n(300))
    R_H = 1.0 / (n * q * 1e6)
    V_H = R_H * 1e3 * B
    fig = go.Figure(layout=make_layout('Hall Voltage vs Magnetic Field'))
    fig.add_trace(go.Scatter(x=B, y=V_H * 1e3, mode='lines',
        line=dict(color=COLORS[4], width=2.5),
        fill='tozeroy', fillcolor='rgba(155,93,229,0.04)'))
    fig.update_layout(xaxis_title='B (T)',
                      yaxis_title='V_H (mV)  [I=1mA, d=1mm]')
    return pplot(fig)


def plt_iv_diode(T, Eg, Nc=None, Nv=None):
    V   = np.linspace(-0.5, 0.8, _n(500))
    Nc  = fb.NC_SI_CM3_300K if Nc is None else Nc
    Nv  = fb.NV_SI_CM3_300K if Nv is None else Nv
    ni  = fb.intrinsic_carrier_concentration(Nc, Nv, Eg, T)
    I0  = q * ni * 1e-4
    I   = I0 * (np.exp(np.clip(q*V/(k_B*max(T,1)), -500, 500)) - 1)
    fig = go.Figure(layout=make_layout('p-n Junction I-V Characteristic'))
    fig.add_trace(go.Scatter(x=V, y=I*1e3, mode='lines',
        line=dict(color=COLORS[1], width=2.5)))
    fig.add_vline(x=0, line=dict(color='rgba(255,255,255,0.07)', width=1))
    fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.07)', width=1))
    fig.update_layout(xaxis_title='V (V)', yaxis_title='I (mA)')
    return pplot(fig)


def plt_schottky(Eg=1.12):
    phi_m = 4.5
    chi   = 4.05
    phi_B = phi_m - chi
    x     = np.linspace(0, 5, _n(400))
    Ec    = -phi_B * np.exp(-x / 1.5)
    Ev    = Ec - Eg
    fig   = go.Figure(layout=make_layout('Schottky Barrier Band Diagram'))
    fig.add_trace(go.Scatter(x=x, y=Ec, mode='lines', name='Ec',
        line=dict(color=COLORS[0], width=2)))
    fig.add_trace(go.Scatter(x=x, y=Ev, mode='lines', name='Ev',
        line=dict(color=COLORS[3], width=2)))
    fig.add_vline(x=0,
                  line=dict(color='rgba(255,255,255,0.13)',
                            dash='dash', width=1),
                  annotation_text='Metal|Semi',
                  annotation_font=dict(color='#6a8aaa', size=8))
    fig.update_layout(xaxis_title='x (nm)', yaxis_title='E (eV)')
    return pplot(fig)


def plt_phonon():
    q_arr   = np.linspace(-np.pi, np.pi, _n(400))
    m1, m2, C = 1.0, 2.0, 1.0
    sq_a    = (C*(m1+m2)/(m1*m2)
               - C/(m1*m2)*np.sqrt((m1+m2)**2
                 - 4*m1*m2*np.sin(q_arr/2)**2))
    sq_o    = (C*(m1+m2)/(m1*m2)
               + C/(m1*m2)*np.sqrt((m1+m2)**2
                 - 4*m1*m2*np.sin(q_arr/2)**2))
    sq_a    = np.clip(sq_a, 0, None)
    sq_o    = np.clip(sq_o, 0, None)
    fig     = go.Figure(layout=make_layout('Phonon Dispersion (1D Diatomic Chain)'))
    fig.add_trace(go.Scatter(x=q_arr, y=np.sqrt(sq_a), mode='lines',
        name='Acoustic', line=dict(color=COLORS[1], width=2)))
    fig.add_trace(go.Scatter(x=q_arr, y=np.sqrt(sq_o), mode='lines',
        name='Optical',  line=dict(color=COLORS[2], width=2)))
    fig.update_layout(xaxis_title='q (units of 1/a)', yaxis_title='omega (arb.)')
    return pplot(fig)


def plt_depletion(Nc, Nd, Eg):
    eps_r = 11.7
    eps   = eps_r * 8.854e-12
    Vbi   = np.linspace(0.1, 1.5, _n(300))
    Na_m  = max(Nc * 1e3, 1e16) * 1e6
    Nd_m  = max(Nd,        1e13) * 1e6
    W     = np.sqrt(2 * eps * Vbi * (Na_m + Nd_m) / (q * Na_m * Nd_m))
    fig   = go.Figure(layout=make_layout('Depletion Width vs Built-in Voltage'))
    fig.add_trace(go.Scatter(x=Vbi, y=W*1e9, mode='lines',
        line=dict(color=COLORS[5], width=2.5),
        fill='tozeroy', fillcolor='rgba(76,201,240,0.04)'))
    fig.update_layout(xaxis_title='Vbi (V)', yaxis_title='W (nm)')
    return pplot(fig)


def plt_ek_compare(Eg=1.12):
    """Schematic E(k): direct gap (Γ–Γ) vs indirect (Γ–Δ) for teaching."""
    k = np.linspace(-1.0, 1.0, _n(360))
    kin = 0.78
    scale_cb = Eg * 0.22
    scale_vb = Eg * 0.18
    Ev_dir = -Eg / 2.0 - scale_vb * k**2
    Ec_dir = Eg / 2.0 + scale_cb * k**2
    Ev_ind = Ev_dir.copy()
    Ec_ind = Eg / 2.0 + scale_cb * 2.2 * (k - kin) ** 2 + 0.06 * Eg

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Direct gap (e.g. GaAs)',
            'Indirect gap (e.g. Si)',
        ),
        horizontal_spacing=0.08,
    )
    for col, Ev, Ec in ((1, Ev_dir, Ec_dir), (2, Ev_ind, Ec_ind)):
        fig.add_trace(
            go.Scatter(x=k, y=Ev, mode='lines', name='Valence band' if col == 1 else None,
                       legendgroup='v', showlegend=(col == 1),
                       line=dict(color=COLORS[3], width=2.5)),
            row=1, col=col,
        )
        fig.add_trace(
            go.Scatter(x=k, y=Ec, mode='lines', name='Conduction band' if col == 1 else None,
                       legendgroup='c', showlegend=(col == 1),
                       line=dict(color=COLORS[0], width=2.5)),
            row=1, col=col,
        )
    fig.update_xaxes(title_text='k (relative to zone edge)', gridcolor='#1a2840',
                     linecolor='#1a2840', tickfont=dict(size=9), zerolinecolor='#1a2840')
    fig.update_yaxes(title_text='Energy (eV)', gridcolor='#1a2840',
                     linecolor='#1a2840', tickfont=dict(size=9), zerolinecolor='#1a2840')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,15,26,1)',
        font=dict(family='Space Mono, monospace', color='#6a8aaa', size=10),
        margin=dict(l=55, r=18, t=56, b=50),
        title=dict(text='Energy band structure E(k)', font=dict(size=11, color='#2e4460'), x=0.02),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1a2840', borderwidth=1,
                    font=dict(size=9)),
    )
    return pplot(fig)


def plt_pn_junction_bands(Eg=1.12):
    """Equilibrium Ec, Ev across a p–n junction (schematic band bending)."""
    x = np.linspace(-3.0, 3.0, _n(480))
    W = 0.38
    tran = 0.5 * (1.0 + np.tanh(x / W))
    Ec_n, Ec_p = 0.02, Eg * 0.65 + 0.15
    Ev_n, Ev_p = -Eg + 0.02, -0.04
    Ec = Ec_n + (Ec_p - Ec_n) * tran
    Ev = Ev_n + (Ev_p - Ev_n) * tran
    Ei = 0.5 * (Ec + Ev)
    Ef = -0.18

    fig = go.Figure(layout=make_layout('P–N junction: built-in field & band bending'))
    fig.add_trace(go.Scatter(x=x, y=Ec, mode='lines', name='Ec',
                             line=dict(color=COLORS[0], width=2.5)))
    fig.add_trace(go.Scatter(x=x, y=Ev, mode='lines', name='Ev',
                             line=dict(color=COLORS[3], width=2.5)))
    fig.add_trace(go.Scatter(x=x, y=Ei, mode='lines', name='Ei (intrinsic)',
                             line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot')))
    fig.add_hline(y=Ef, line=dict(color=COLORS[2], dash='dash', width=1.4),
                  annotation_text='Ef (equilibrium)', annotation_font=dict(size=8, color='#ffb800'))
    fig.add_vline(x=0, line=dict(color='rgba(255,255,255,0.09)', width=1))
    fig.update_layout(xaxis_title='Position x (arb. units, n-side ← | → p-side)',
                      yaxis_title='Energy (eV)')
    return pplot(fig)


def plt_dos_qw_2d(m_eff_ratio, L_nm=12.0):
    """2D density of states staircase for a quantum well (particle-in-a-box subbands)."""
    m = max(float(m_eff_ratio), 0.01) * m0
    L = max(float(L_nm) * 1e-9, 5e-10)
    n_max = 10
    ns = np.arange(1, n_max + 1, dtype=float)
    En = (hbar * np.pi * ns / L) ** 2 / (2.0 * m) / eV
    emax = max(float(En[-1]) * 1.15, 0.2)
    E = np.linspace(-0.02, emax, _n(900))
    const = m / (np.pi * hbar ** 2) * 1e-17
    dos = np.zeros_like(E)
    for e_n in En:
        dos += (E >= e_n).astype(float) * const

    fig = go.Figure(layout=make_layout('2D density of states in a quantum well (staircase)'))
    fig.add_trace(go.Scatter(
        x=E, y=dos, mode='lines', name='g2D(E)',
        line=dict(color=COLORS[1], width=2.2),
        line_shape='hv'))
    for i, e_n in enumerate(En):
        fig.add_vline(x=e_n, line=dict(color='rgba(0,240,255,0.25)', width=1, dash='dot'),
                      annotation_text=f'n={int(ns[i])}' if i < 4 else None,
                      annotation_font=dict(size=7, color='#6a8aaa'))
    fig.update_layout(xaxis_title='E above well bottom (eV)', yaxis_title='g2D (scaled, arb.)')
    return pplot(fig)


def plt_tauc(Eg=1.12):
    """Tauc plot: (αhν)² vs hν for a direct allowed transition (illustrative)."""
    hnu = np.linspace(max(0.1, Eg - 0.5), Eg + 2.2, _n(500))
    A = 1.2e5
    root = np.sqrt(np.maximum(hnu - Eg, 0.0))
    alpha = A * root / np.maximum(hnu, 1e-6)
    alpha += 180.0 * np.exp((hnu - Eg) / 0.06) * (hnu < Eg)
    y_tauc = (alpha * hnu) ** 2

    fig = go.Figure(layout=make_layout('Tauc plot — (α·hν)² vs photon energy (direct gap)'))
    fig.add_trace(go.Scatter(x=hnu, y=y_tauc, mode='lines', name='(α·hν)²',
                             line=dict(color=COLORS[4], width=2.2),
                             fill='tozeroy', fillcolor='rgba(155,93,229,0.06)'))
    fig.add_vline(x=Eg, line=dict(color='rgba(255,184,0,0.45)', dash='dash', width=1),
                  annotation_text=f'Eg ≈ {Eg:.2f} eV', annotation_font=dict(size=8, color='#ffb800'))
    fig.update_layout(xaxis_title='hν (eV)', yaxis_title='(α·hν)² (arb.)')
    return pplot(fig)


def plt_sdh():
    """Illustrative Shubnikov–de Haas: longitudinal resistance oscillations vs B."""
    B = np.linspace(0.2, 8.0, _n(700))
    B_safe = np.maximum(B, 0.08)
    F = 52.0
    dingle = np.exp(-0.18 / B_safe)
    osc = 0.14 * dingle * np.cos(2.0 * np.pi * F / B_safe + 0.25)
    R = 1200.0 * (1.0 + osc)

    fig = go.Figure(layout=make_layout('Shubnikov–de Haas (illustrative R_xx vs B)'))
    fig.add_trace(go.Scatter(
        x=B, y=R, mode='lines', name='R_xx',
        line=dict(color=COLORS[5], width=2.0),
        hovertemplate='B=%{x:.2f} T<br>R=%{y:.1f}<extra></extra>'))
    fig.update_layout(
        xaxis_title='B (T)', yaxis_title='R_xx (Ω, model)',
        yaxis=dict(rangemode='tozero'))
    return pplot(fig)


# Routes

@app.route('/', methods=['GET', 'POST'])
def home():
    # Defaults — Si literature values (see fermi_boltzmann module docstring)
    Nc          = fb.NC_SI_CM3_300K
    Nv          = fb.NV_SI_CM3_300K
    Eg          = fb.EG_SI_EV_300K
    Nd          = 1e17
    T           = 300.0
    tau         = 0.24e-15
    vF          = 1e6
    m_eff_ratio = 0.26
    a           = 5e-10
    V0          = 10.0
    b           = 2e-10
    bz_lattice  = 'square'
    bz_a        = 1.0
    bz_b        = 1.5
    bz_angle    = 120.0
    bz_zones    = 10

    active_section = (request.args.get('section') or '').strip()
    if request.method == 'POST':
        active_section = (request.form.get('active_section') or active_section).strip()
    if active_section not in oqs.SECTION_BY_ID:
        active_section = ''

    recip_vertices = rl.DEFAULT_VERTICES.copy()
    recip_n = int(recip_vertices.shape[0])
    recip_c_axis = 0.6

    if request.method == 'POST':
        Nc          = gf(request.form, 'Nc',    Nc)
        Nv          = gf(request.form, 'Nv',    Nv)
        Eg          = gf(request.form, 'Eg',    Eg)
        Nd          = gf(request.form, 'Nd',    Nd)
        T           = gf(request.form, 'T',     T)
        tau         = gf(request.form, 'tau',   tau)
        vF          = gf(request.form, 'vF',    vF)
        m_eff_ratio = gf(request.form, 'm_eff', m_eff_ratio)
        a           = gf(request.form, 'a',     a)
        V0          = gf(request.form, 'V0',    V0)
        b           = gf(request.form, 'b',     b)
        _lat = request.form.get('bz_lattice', bz_lattice)
        bz_lattice  = _lat if _lat in BZ_LATTICES else bz_lattice
        bz_a        = gf(request.form, 'bz_a',     bz_a)
        bz_b        = gf(request.form, 'bz_b',     bz_b)
        bz_angle    = gf(request.form, 'bz_angle', bz_angle)
        bz_zones    = 10
        bz_layers   = parse_bz_layers_raw(
            request.form.get('bz_layers'), bz_zones)
        recip_vertices, recip_n, recip_c_axis = parse_recip_from_form(request.form)
        open_plots = parse_open_plots(request.form)
    else:
        open_plots = frozenset()
        bz_layers = (1, 2, 3, 4)

    section = oqs.SECTION_BY_ID.get(active_section)
    sidebar_tags = oqs.sidebar_tags_for_section(active_section)
    section_plots = oqs.plots_for_section(active_section)
    wiki_note = fetch_wikipedia_extract(section['wiki_title']) if section else ''
    theory_block = ''
    if section:
        theory_block = section['theory']
        if wiki_note:
            theory_block += ' — Wikipedia: ' + wiki_note

    recip_list = recip_vertices.tolist()
    while len(recip_list) < 8:
        recip_list.append([0.0, float(len(recip_list)) * 0.2, 0.0])

    ctx = simulation_context(
        Nc, Nv, Eg, Nd, T, tau, vF, m_eff_ratio, a, V0, b,
        bz_lattice, bz_a, bz_b, bz_angle, bz_zones, bz_layers=bz_layers,
        recip_vertices=recip_vertices, recip_c_axis=recip_c_axis,
    )
    ni = ctx['ni']
    Ef = ctx['Ef']
    n, pp = ctx['n'], ctx['p']
    mu = ctx['mu']
    sigma = ctx['sigma']
    l_mfp = ctx['l_mfp']

    def build_plots():
        out = {}
        for key in PLOT_KEYS:
            if key in open_plots:
                out[key] = render_plot_html(key, ctx)
            else:
                out[key] = PLOT_PLACEHOLDER
        return out

    plots = build_plots()
    open_plots_csv = ','.join(sorted(open_plots))
    bz_layers_all = tuple(range(1, 11)) == tuple(bz_layers)

    return render_template(
        'index.html',
        Nc=Nc, Nv=Nv, Eg=Eg, Nd=Nd, T=T,
        tau=tau, vF=vF, m_eff=m_eff_ratio,
        a=a, V0=V0, b=b,
        ni=ni, Ef=Ef, n=n, p=pp,
        mu=mu, sigma=sigma, l=l_mfp,
        bz_lattice=bz_lattice, bz_a=bz_a, bz_b=bz_b,
        bz_angle=bz_angle, bz_zones=bz_zones,
        bz_layers=bz_layers,
        bz_layers_all=bz_layers_all,
        bz_layers_csv=('all' if bz_layers_all else ','.join(str(z) for z in bz_layers)),
        open_plots=open_plots,
        open_plots_csv=open_plots_csv,
        plot_placeholder_html=PLOT_PLACEHOLDER,
        active_section=active_section,
        oq_sections=oqs.SECTIONS_ORDER,
        section_meta=section,
        sidebar_tags=sidebar_tags,
        section_plots=section_plots,
        theory_block=theory_block,
        plot_titles=PLOT_TITLES,
        recip_n=recip_n,
        recip_c_axis=recip_c_axis,
        recip_list=recip_list,
        recip_vertices=recip_vertices,
        **plots
    )


@app.route('/api/bz', methods=['POST'])
def api_bz():
    data = request.get_json() or {}
    raw_lat = data.get('lattice', 'square')
    lattice = raw_lat if raw_lat in BZ_LATTICES else 'square'
    try:
        a = float(data.get('a', 1.0))
        b = float(data.get('b', 1.5))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid lattice constants'}), 400
    try:
        angle = float(data.get('angle', 120.0))
    except (TypeError, ValueError):
        angle = 120.0
    # Full extended-zone geometry up to 10th BZ; layers choose what to draw.
    n_zones = 10
    layer_src = data.get('layers')
    if layer_src is None:
        layer_src = data.get('bz_layers')
    zones_to_show = parse_bz_layers_json(layer_src, n_zones)

    try:
        div = plot_brillouin_zones(
            lattice=lattice, a=a, b=b,
            angle=angle, n_zones=n_zones,
            zones_to_show=zones_to_show)
        return jsonify({'plot': div})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/plot', methods=['POST'])
def api_plot():
    data = request.get_json(silent=True) or {}
    plot_key = data.get('plot')
    if plot_key not in PLOT_KEYS:
        return jsonify({'error': 'Invalid plot key'}), 400
    try:
        ctx = context_from_api_payload(data)
        html = render_plot_html(plot_key, ctx)
    except (TypeError, ValueError) as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify({'plot': html})


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/about')
def about():
    return render_template('summary.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Render sets RENDER=true; the Flask stat reloader confuses port/health checks there.
    on_render = os.environ.get('RENDER', '').lower() == 'true'
    _fd = os.environ.get('FLASK_DEBUG', '1' if not on_render else '0').lower()
    debug = _fd in ('1', 'true', 'yes')
    app.run(host='0.0.0.0', port=port, debug=debug, use_reloader=(debug and not on_render))
