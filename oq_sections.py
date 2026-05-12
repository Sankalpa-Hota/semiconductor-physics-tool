# OPEN-Quantum section definitions (guided exploration + plot grouping)
#
# Each section lists Plotly keys from main.PLOT_KEYS. Sidebar panel tags:
#   p_stats   — Nc, Nv, Eg, Nd, T (carrier / junction shared)
#   p_transport — Drude τ, vF, m*
#   p_kp      — Kronig–Penney a, V0, b
#   p_bz      — (controls live in main column for BZ)
#   p_lattice — reciprocal vertex inputs

SECTIONS_ORDER = [
    {
        'id': 'stats',
        'title': 'Quantum statistics & occupation',
        'subtitle': 'Fermi–Dirac statistics, density of states, quantum-well DOS',
        'sidebar': frozenset({'p_stats'}),
        'plots': ('fd', 'dos', 'dos_qw'),
        'wiki_title': 'Fermi–Dirac statistics',
        'theory': (
            'Electrons in semiconductors obey Fermi–Dirac occupation f(E). '
            'The conduction-band density of states g(E) sets how many states are available; '
            'in a quantum well, motion is quantized and g(E) becomes a staircase of subbands.'
        ),
    },
    {
        'id': 'transport',
        'title': 'Transport & carrier motion',
        'subtitle': 'Intrinsic concentration, mobility, conductivity, quantum oscillations',
        'sidebar': frozenset({'p_stats', 'p_transport'}),
        'plots': ('ni_T', 'mu_T', 'carr_T', 'cond_T', 'sdh'),
        'wiki_title': 'Electron mobility',
        'theory': (
            'Carrier transport combines band occupation (n, p) with scattering-limited mobility μ. '
            'Matthiessen’s rule approximates combined lattice and ionized-impurity scattering; '
            'Shubnikov–de Haas oscillations (illustrative here) reflect Landau quantization in strong B.'
        ),
    },
    {
        'id': 'doping',
        'title': 'Doping & energy levels',
        'subtitle': 'Fermi level, band diagram, resistivity, Varshni gap, Tauc plot',
        'sidebar': frozenset({'p_stats'}),
        'plots': ('ef_dop', 'band', 'rho_dop', 'eg_T', 'tauc'),
        'wiki_title': 'Doping (semiconductor)',
        'theory': (
            'Adding donors or acceptors pins the Fermi level relative to Ec and Ev, shifting '
            'equilibrium carrier densities. Temperature reduces the bandgap (Varshni) and changes '
            'resistivity; optical measurements often use a Tauc plot to infer an effective gap.'
        ),
    },
    {
        'id': 'devices',
        'title': 'Junctions & devices',
        'subtitle': 'p–n diode, Schottky, Hall effect, depletion, band bending',
        'sidebar': frozenset({'p_stats'}),
        'plots': ('iv', 'schot', 'hall', 'depl', 'pn_x'),
        'wiki_title': 'P-n junction',
        'theory': (
            'A p–n junction establishes a built-in field and rectifying I–V characteristics. '
            'Schottky metal–semiconductor contacts show barrier lowering; the Hall effect probes '
            'carrier sign and density; depletion widths follow from Poisson’s equation.'
        ),
    },
    {
        'id': 'ek',
        'title': 'Band structure E(k)',
        'subtitle': 'Direct vs indirect gap in k-space',
        'sidebar': frozenset({'p_stats'}),
        'plots': ('ek_comp',),
        'wiki_title': 'Electronic band structure',
        'theory': (
            'Near band edges, E(k) is often parabolic with effective mass. A direct gap places '
            'the conduction-band minimum at the same k as the valence-band maximum; an indirect '
            'gap (e.g. Si) offsets them, affecting light emission efficiency.'
        ),
    },
    {
        'id': 'kp',
        'title': 'Kronig–Penney model',
        'subtitle': 'Periodic potential → allowed bands and gaps',
        'sidebar': frozenset({'p_kp'}),
        'plots': ('kp1d', 'kp2d', 'kp3d'),
        'wiki_title': 'Kronig-Penney model',
        'theory': (
            'The one-dimensional Kronig–Penney model solves a square periodic barrier, showing '
            'how Bragg-like conditions open energy gaps and fold dispersion into the reduced zone. '
            'It is a minimal quantum picture of Bloch bands in crystals.'
        ),
    },
    {
        'id': 'bz',
        'title': 'Brillouin zones',
        'subtitle': 'Wigner–Seitz zones in k-space — choose zone index 1 … 10',
        'sidebar': frozenset({'p_bz'}),
        'plots': ('bz_plot',),
        'wiki_title': 'Brillouin zone',
        'theory': (
            'The first Brillouin zone is the Wigner–Seitz cell of the reciprocal lattice. '
            'Higher zones tile k-space with equal area; electron dynamics and diffraction are '
            'often phrased in terms of zone boundaries (Bragg planes).'
        ),
    },
    {
        'id': 'lattice',
        'title': 'Lattice dynamics & reciprocal space',
        'subtitle': 'Phonons, custom real-space vertices → primitive vectors & bⱼ',
        'sidebar': frozenset({'p_lattice'}),
        'plots': ('phonon', 'recip'),
        'wiki_title': 'Reciprocal lattice',
        'theory': (
            'Phonons are quantized lattice vibrations; the dispersion ω(q) encodes acoustic and '
            'optical branches. Primitive vectors a₁,a₂,a₃ span the direct cell; reciprocals '
            'bⱼ satisfy aᵢ·bⱼ=2πδᵢⱼ and underpin k-space band structure and diffraction.'
        ),
    },
]

SECTION_BY_ID = {s['id']: s for s in SECTIONS_ORDER}


def sidebar_tags_for_section(section_id):
    if not section_id:
        return frozenset()
    s = SECTION_BY_ID.get(section_id)
    return s['sidebar'] if s else frozenset()


def plots_for_section(section_id):
    s = SECTION_BY_ID.get(section_id)
    return s['plots'] if s else ()
