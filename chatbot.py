"""
chatbot.py — Evergreen RAG Chatbot for Semiconductor Physics
"""

from flask import Blueprint, request, jsonify, render_template
import json, re, os, math, time, hashlib, threading
import urllib.request, urllib.parse, urllib.error
from collections import defaultdict

chatbot_bp = Blueprint('chatbot', __name__)

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
CACHE_TTL    = 3600          # seconds before re-fetching a source
MAX_CHUNKS   = 6             # top-k chunks to inject into prompt
CHUNK_SIZE   = 400           # words per chunk

# ── Source catalogue (evergreen — APIs + stable reference URLs) ─
TOPIC_SOURCES = {
    "fermi_dirac": [
        "https://en.wikipedia.org/w/api.php?action=query&titles=Fermi%E2%80%93Dirac_statistics&prop=extracts&exintro=false&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Fermi_level&prop=extracts&exintro=false&explaintext=true&format=json",
    ],
    "band_theory": [
        "https://en.wikipedia.org/w/api.php?action=query&titles=Electronic_band_structure&prop=extracts&exintro=false&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Kronig%E2%80%93Penney_model&prop=extracts&exintro=false&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Nearly_free_electron_model&prop=extracts&explaintext=true&format=json",
    ],
    "brillouin": [
        "https://en.wikipedia.org/w/api.php?action=query&titles=Brillouin_zone&prop=extracts&exintro=false&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Wigner%E2%80%93Seitz_cell&prop=extracts&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Reciprocal_lattice&prop=extracts&explaintext=true&format=json",
    ],
    "carrier_transport": [
        "https://en.wikipedia.org/w/api.php?action=query&titles=Drude_model&prop=extracts&exintro=false&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Electron_mobility&prop=extracts&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Hall_effect&prop=extracts&explaintext=true&format=json",
    ],
    "doping": [
        "https://en.wikipedia.org/w/api.php?action=query&titles=Semiconductor_doping&prop=extracts&exintro=false&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Extrinsic_semiconductor&prop=extracts&explaintext=true&format=json",
    ],
    "pn_junction": [
        "https://en.wikipedia.org/w/api.php?action=query&titles=P%E2%80%93n_junction&prop=extracts&exintro=false&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Shockley_diode_equation&prop=extracts&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Depletion_region&prop=extracts&explaintext=true&format=json",
    ],
    "mosfet": [
        "https://en.wikipedia.org/w/api.php?action=query&titles=MOSFET&prop=extracts&exintro=false&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Threshold_voltage&prop=extracts&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Short-channel_effect&prop=extracts&explaintext=true&format=json",
    ],
    "cmos": [
        "https://en.wikipedia.org/w/api.php?action=query&titles=CMOS&prop=extracts&exintro=false&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Logic_gate&prop=extracts&explaintext=true&format=json",
    ],
    "phonons": [
        "https://en.wikipedia.org/w/api.php?action=query&titles=Phonon&prop=extracts&exintro=false&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Debye_model&prop=extracts&explaintext=true&format=json",
    ],
    "density_of_states": [
        "https://en.wikipedia.org/w/api.php?action=query&titles=Density_of_states&prop=extracts&exintro=false&explaintext=true&format=json",
    ],
    "semiconductors_general": [
        "https://en.wikipedia.org/w/api.php?action=query&titles=Semiconductor&prop=extracts&exintro=false&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Intrinsic_semiconductor&prop=extracts&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Band_gap&prop=extracts&explaintext=true&format=json",
    ],
    "mos_capacitor": [
        "https://en.wikipedia.org/w/api.php?action=query&titles=MOS_capacitor&prop=extracts&exintro=false&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Metal%E2%80%93oxide%E2%80%93semiconductor&prop=extracts&explaintext=true&format=json",
    ],
    "schottky": [
        "https://en.wikipedia.org/w/api.php?action=query&titles=Schottky_barrier&prop=extracts&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Schottky_diode&prop=extracts&explaintext=true&format=json",
    ],
    "heterostructures": [
        "https://en.wikipedia.org/w/api.php?action=query&titles=Heterojunction&prop=extracts&explaintext=true&format=json",
        "https://en.wikipedia.org/w/api.php?action=query&titles=Quantum_well&prop=extracts&explaintext=true&format=json",
    ],
}

# ArXiv query templates for research-level questions
ARXIV_TOPICS = {
    "semiconductor": "semiconductor+physics+tutorial",
    "mosfet":        "MOSFET+short+channel+effects",
    "band":          "electronic+band+structure+semiconductor",
    "quantum":       "quantum+well+semiconductor+physics",
    "phonon":        "phonon+scattering+semiconductor",
}


# ══════════════════════════════════════════════════════════════
#  CACHE  (in-memory, thread-safe, TTL-based)
# ══════════════════════════════════════════════════════════════
_cache_lock = threading.Lock()
_cache: dict = {}   # key → {text, ts}


def cache_get(key: str):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry['ts']) < CACHE_TTL:
            return entry['text']
    return None


def cache_set(key: str, text: str):
    with _cache_lock:
        _cache[key] = {'text': text, 'ts': time.time()}


# ══════════════════════════════════════════════════════════════
#  FETCHERS
# ══════════════════════════════════════════════════════════════

def http_get(url: str, timeout=10) -> str:
    """Safe HTTP GET, returns empty string on failure."""
    key = hashlib.md5(url.encode()).hexdigest()
    cached = cache_get(key)
    if cached:
        return cached
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SemiconductorPhysicsBot/2.0 (educational tool)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
            cache_set(key, raw)
            return raw
    except Exception:
        return ""


def fetch_wikipedia(api_url: str) -> str:
    """Extract plain text from Wikipedia API JSON response."""
    raw = http_get(api_url)
    if not raw:
        return ""
    try:
        data = json.loads(raw)
        pages = data.get("query", {}).get("pages", {})
        texts = []
        for page in pages.values():
            extract = page.get("extract", "")
            if extract:
                texts.append(extract)
        return "\n\n".join(texts)
    except Exception:
        return ""


def fetch_arxiv(query: str, max_results=3) -> str:
    """Fetch abstracts from ArXiv search API."""
    encoded = urllib.parse.quote(query)
    url = (f"https://export.arxiv.org/api/query?"
           f"search_query=all:{encoded}&start=0&max_results={max_results}"
           f"&sortBy=relevance&sortOrder=descending")
    raw = http_get(url, timeout=12)
    if not raw:
        return ""
    # Extract titles + summaries from Atom XML
    titles   = re.findall(r'<title>(.*?)</title>', raw, re.DOTALL)[1:]
    summaries = re.findall(r'<summary>(.*?)</summary>', raw, re.DOTALL)
    result = []
    for t, s in zip(titles, summaries):
        t = re.sub(r'\s+', ' ', t).strip()
        s = re.sub(r'\s+', ' ', s).strip()
        result.append(f"[ArXiv] {t}\n{s}")
    return "\n\n".join(result)


# ══════════════════════════════════════════════════════════════
#  QUERY → TOPIC MAPPING
# ══════════════════════════════════════════════════════════════

TOPIC_KEYWORDS = {
    "fermi_dirac":         ["fermi", "dirac", "distribution", "f(e)", "occupation", "chemical potential"],
    "band_theory":         ["band", "kronig", "penney", "gap", "allowed", "forbidden", "dispersion", "effective mass"],
    "brillouin":           ["brillouin", "bz", "zone", "wigner", "seitz", "reciprocal", "γ", "gamma", "k-space"],
    "carrier_transport":   ["drude", "mobility", "conductivity", "drift", "diffusion", "transport", "hall", "mean free", "scattering"],
    "doping":              ["doping", "dopant", "n-type", "p-type", "donor", "acceptor", "phosphorus", "boron", "extrinsic"],
    "pn_junction":         ["p-n", "pn", "junction", "diode", "depletion", "shockley", "built-in", "biasing", "forward", "reverse"],
    "mosfet":              ["mosfet", "transistor", "threshold", "vth", "channel", "drain", "source", "gate", "inversion", "dibl", "saturation"],
    "cmos":                ["cmos", "inverter", "nmos", "pmos", "logic", "nand", "nor", "pull-up", "pull-down", "power dissipation"],
    "phonons":             ["phonon", "acoustic", "optical", "lattice vibration", "debye", "einstein", "dispersion"],
    "density_of_states":   ["density of states", "dos", "d.o.s", "van hove", "singularity", "parabolic"],
    "semiconductors_general": ["semiconductor", "silicon", "germanium", "gaas", "intrinsic", "ni", "bandgap", "crystal", "lattice"],
    "mos_capacitor":       ["mos", "moscap", "capacitor", "accumulation", "depletion", "inversion", "band bending", "oxide"],
    "schottky":            ["schottky", "metal", "barrier", "work function", "electron affinity"],
    "heterostructures":    ["hetero", "heterojunction", "quantum well", "2deg", "type i", "type ii", "anderson"],
}


def identify_topics(query: str) -> list:
    """Return list of relevant topic keys sorted by keyword match count."""
    q = query.lower()
    scores = defaultdict(int)
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                scores[topic] += 1
    if scores:
        return [t for t, _ in sorted(scores.items(), key=lambda x: -x[1])[:4]]
    # Default: return broad topics
    return ["semiconductors_general", "band_theory", "carrier_transport"]


# ══════════════════════════════════════════════════════════════
#  TEXT CHUNKING
# ══════════════════════════════════════════════════════════════

def chunk_text(text: str, size=CHUNK_SIZE) -> list:
    """Split text into overlapping word-chunks."""
    words = text.split()
    chunks = []
    step = max(size - 80, 100)   # 80-word overlap
    for i in range(0, max(len(words) - 10, 1), step):
        chunk = " ".join(words[i:i+size])
        if len(chunk.strip()) > 60:
            chunks.append(chunk)
    return chunks


# ══════════════════════════════════════════════════════════════
#  TF-IDF RANKER  (no external deps)
# ══════════════════════════════════════════════════════════════

def tokenize(text: str) -> list:
    return re.findall(r'[a-zA-Z0-9\u03b1-\u03c9\u0391-\u03a9]{2,}', text.lower())


def tfidf_score(query_tokens: list, chunk: str, corpus_size: int,
                doc_freq: dict) -> float:
    """Compute TF-IDF cosine similarity between query and chunk."""
    chunk_tokens = tokenize(chunk)
    if not chunk_tokens:
        return 0.0
    tf = defaultdict(float)
    for tok in chunk_tokens:
        tf[tok] += 1.0
    norm = len(chunk_tokens)
    score = 0.0
    for tok in set(query_tokens):
        if tok in tf:
            tf_val  = tf[tok] / norm
            df      = doc_freq.get(tok, 1)
            idf_val = math.log((corpus_size + 1) / (df + 1)) + 1
            score  += tf_val * idf_val
    return score


def rank_chunks(query: str, chunks: list) -> list:
    """Return top-k chunks by TF-IDF relevance to query."""
    if not chunks:
        return []
    q_tokens = tokenize(query)
    # Build doc freq over corpus
    doc_freq = defaultdict(int)
    for c in chunks:
        for tok in set(tokenize(c)):
            doc_freq[tok] += 1
    scored = [(tfidf_score(q_tokens, c, len(chunks), doc_freq), c)
              for c in chunks]
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:MAX_CHUNKS] if _ > 0]


# ══════════════════════════════════════════════════════════════
#  RETRIEVAL PIPELINE
# ══════════════════════════════════════════════════════════════

def retrieve(query: str) -> str:
    """
    Full retrieval pipeline:
    1. Identify relevant topics from query
    2. Fetch Wikipedia articles for those topics (cached)
    3. Fetch ArXiv abstracts for research flavor
    4. Chunk all text
    5. TF-IDF rank and return top chunks
    """
    topics = identify_topics(query)
    all_text_parts = []

    # Wikipedia
    for topic in topics:
        urls = TOPIC_SOURCES.get(topic, [])
        for url in urls[:2]:  # max 2 Wikipedia pages per topic
            text = fetch_wikipedia(url)
            if text:
                all_text_parts.append(f"[Wikipedia/{topic}]\n{text}")

    # ArXiv — pick most relevant topic for research context
    primary = topics[0] if topics else "semiconductor"
    arxiv_q = ARXIV_TOPICS.get(primary, f"semiconductor+{primary}")
    arxiv_text = fetch_arxiv(arxiv_q, max_results=2)
    if arxiv_text:
        all_text_parts.append(f"[ArXiv Recent Research]\n{arxiv_text}")

    if not all_text_parts:
        return ""

    # Chunk
    all_chunks = []
    for part in all_text_parts:
        all_chunks.extend(chunk_text(part))

    # Rank
    top_chunks = rank_chunks(query, all_chunks)
    return "\n\n---\n\n".join(top_chunks)


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT BUILDER  (dynamic, uses retrieved context)
# ══════════════════════════════════════════════════════════════

STATIC_PREAMBLE = """You are PhysBot, an expert AI assistant for semiconductor physics,
built into an interactive simulation tool. You have access to live-retrieved knowledge
from Wikipedia and ArXiv papers.

Your role:
- Answer questions about semiconductor physics precisely and clearly
- Use Unicode equations: ħ, μ, σ, ε, Γ, π, √, ²
- Reference the retrieved context below; synthesize don't just copy
- For numerical questions: state the formula, define variables, compute
- Connect theory to the simulation plots visible in the tool
- Cite [Wikipedia] or [ArXiv] when using retrieved content
- Keep answers focused: 2-5 paragraphs, equations on their own lines
- If asked about a material, include typical parameter values
"""


def build_prompt(query: str, context: str, history: list) -> list:
    """Build message list for Ollama chat API."""
    system_content = STATIC_PREAMBLE
    if context:
        system_content += f"\n\n=== RETRIEVED KNOWLEDGE (live, {time.strftime('%Y-%m-%d')}) ===\n\n{context}"

    messages = [{"role": "system", "content": system_content}]

    for turn in history[-8:]:  # last 8 turns
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": query})
    return messages


# ══════════════════════════════════════════════════════════════
#  OLLAMA INTERFACE
# ══════════════════════════════════════════════════════════════

def query_ollama(messages: list) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.65,
            "top_p": 0.9,
            "num_ctx": 4096,
        }
    }).encode('utf-8')
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        return data["message"]["content"]


# ══════════════════════════════════════════════════════════════
#  FALLBACK ENGINE  (physics-accurate, formula-rich)
# ══════════════════════════════════════════════════════════════

FALLBACK_DB = {
    r"fermi.{0,20}dirac|f\(e\)|occupation|fermi level": """
**Fermi-Dirac Distribution** [Wikipedia/Fermi-Dirac statistics]

f(E) = 1 / (exp((E − Eᶠ) / kT) + 1)

The probability that a quantum state at energy E is occupied by an electron at
temperature T. At T=0K, f(E)=1 for E<Eᶠ and f(E)=0 for E>Eᶠ — a perfect step.
Thermal smearing occurs over ~4kT around Eᶠ.

For semiconductors: Eᶠ lies in the bandgap.
- Intrinsic: Eᶠ ≈ (Ec+Ev)/2 + (kT/2)·ln(Nv/Nc)
- n-type: Eᶠ = Ec + kT·ln(Nd/Nc) → moves toward Ec
- p-type: Eᶠ = Ev − kT·ln(Na/Nv) → moves toward Ev

The intrinsic concentration nᵢ = √(Nc·Nv)·exp(−Eg/2kT), where Nc = 2(2πm*kT/h²)^(3/2).
""",

    r"brillouin|bz|wigner|seitz|k.space|zone": """
**Brillouin Zones** [Wikipedia/Brillouin zone]

The nth Brillouin zone = set of k-points reached from Γ(0,0) by crossing
exactly n−1 Bragg planes. Each zone has equal area = area of reciprocal unit cell.

Construction (Wigner-Seitz method):
1. Draw all reciprocal lattice vectors G from origin
2. For each G, draw its perpendicular bisector plane
3. Zone 1 = innermost region; Zone n = nth shell

Square lattice (spacing a):
- Reciprocal vectors: b₁ = (2π/a)x̂, b₂ = (2π/a)ŷ
- 1st BZ: square |kx| < π/a, |ky| < π/a
- High-symmetry: Γ(0,0), X(π/a,0), M(π/a,π/a)

Hexagonal lattice (spacing a):
- 1st BZ: regular hexagon, radius = 4π/3a
- High-symmetry: Γ, K(4π/3a,0), M(π/a, π/√3a)

Zone folding: all zones map onto 1st BZ in reduced zone scheme.
""",

    r"kronig|penney|band gap|bandgap|band structure|band theory": """
**Kronig-Penney Model & Band Theory** [Wikipedia/Kronig-Penney model]

The Kronig-Penney model solves Schrödinger's equation for a 1D periodic potential
of rectangular barriers (height V₀, width b) in wells (width a−b).

Dispersion relation:
cos(ka) = cos(α(a−b))·cosh(βb) + (β²−α²)/(2αβ)·sin(α(a−b))·sinh(βb)

where α = √(2mE)/ħ,  β = √(2m(V₀−E))/ħ

|RHS| ≤ 1 defines **allowed bands**; |RHS| > 1 gives **forbidden gaps**.
- Larger V₀ → wider gaps
- Larger b/a → stronger gaps
- At zone boundary k=nπ/a: standing waves, energy gaps open

Effective mass: m* = ħ²/(d²E/dk²) — positive at band bottom, negative near top.
""",

    r"drude|mobility|conductiv|transport|drift|mean free": """
**Drude Model & Transport** [Wikipedia/Drude model]

σ = neμ = ne²τ/m*,    μ = eτ/m*,    l = vF·τ

Matthiessen's Rule:   1/μ_total = 1/μ_phonon + 1/μ_impurity

- μ_phonon ∝ T^(−3/2): more phonons at high T scatter electrons more
- μ_impurity ∝ T^(3/2)/Nd: electrons move faster at high T, screen impurities

Silicon at 300K: μn = 1350 cm²/V·s,  μp = 450 cm²/V·s
GaAs at 300K:   μn = 8500 cm²/V·s  (direct gap, lighter m*)

Hall effect:  R_H = 1/(nq) for n-type,  R_H = −1/(pq) for p-type
Hall mobility: μ_H = R_H · σ
""",

    r"dop|n.type|p.type|donor|acceptor|extrinsic|phosphor|boron": """
**Semiconductor Doping** [Wikipedia/Semiconductor doping]

**n-type**: Group V dopants (P, As, Sb in Si) donate one electron.
Nd >> ni → n ≈ Nd,  p ≈ ni²/Nd,  Eᶠ → Ec

**p-type**: Group III dopants (B, Al, Ga in Si) accept one electron (donate hole).
Na >> ni → p ≈ Na,  n ≈ ni²/Na,  Eᶠ → Ev

Charge neutrality: n + Na⁻ = p + Nd⁺
Mass-action law: n·p = ni² (at thermal equilibrium, always)

Compensation: if both donors and acceptors present,
n ≈ (Nd − Na)/2 + √((Nd−Na)²/4 + ni²)

Si at 300K: ni ≈ 1.5×10¹⁰ cm⁻³. A doping of 10¹⁵ cm⁻³ gives n/ni ≈ 67,000×.
""",

    r"p.n|pn junction|diode|depletion|shockley|built.in|biasing": """
**p-n Junction** [Wikipedia/p-n junction]

Built-in voltage:  Vbi = (kT/q)·ln(Na·Nd/ni²)

Depletion width:   W = √(2ε·Vbi·(Na+Nd) / (q·Na·Nd))
  xn = W·Na/(Na+Nd),   xp = W·Nd/(Na+Nd)

Peak electric field: ℰmax = qNd·xn/ε = 2Vbi/W

Shockley equation:  I = I₀·(e^(qV/nkT) − 1)
  I₀ = Aqni²·(Dp/(Lp·Na) + Dn/(Ln·Nd))
  n = 1 (ideal), n = 2 (recombination current dominant)

Forward bias: reduces barrier, exponential current.
Reverse bias: widens depletion, tiny I₀ flows.
Breakdown: Zener (tunneling, low V) or avalanche (impact ionization, high V).
""",

    r"mosfet|mos fet|transistor|threshold|channel|inversion|dibl": """
**MOSFET Physics** [Wikipedia/MOSFET]

Four terminals: Gate (G), Source (S), Drain (D), Body (B)

Threshold voltage:  Vth = Vfb + 2φF + Qd/Cox
  φF = (kT/q)·ln(Na/ni),   Cox = ε_ox/t_ox

Operating regions (NMOS, Vgs > Vth):
- Linear:      Ids = μn·Cox·(W/L)·[(Vgs−Vth)·Vds − Vds²/2]
- Saturation:  Ids = ½·μn·Cox·(W/L)·(Vgs−Vth)²  (Vds > Vgs−Vth)

Short-channel effects (L < ~100nm):
- DIBL: drain field lowers source barrier → reduced Vth at high Vds
- Velocity saturation: v_sat ~ 10⁷ cm/s in Si limits current
- Channel length modulation: λ parameter, Ids ∝ (1+λVds) in saturation
- Subthreshold slope: S = (kT/q)·ln(10)·(1 + Cd/Cox) ≥ 60 mV/dec at 300K
""",

    r"cmos|inverter|nmos|pmos|logic gate|pull.up|pull.down|power": """
**CMOS Logic** [Wikipedia/CMOS]

CMOS inverter: PMOS (pull-up) + NMOS (pull-down) in series between VDD and GND.
- Vin=0 (logic 0): PMOS on, NMOS off → Vout=VDD (logic 1)
- Vin=VDD (logic 1): NMOS on, PMOS off → Vout=0 (logic 0)

Power dissipation:
- Dynamic: Pdyn = α·CL·VDD²·f  (switching activity α, load CL, frequency f)
- Static/leakage: Pstat = Ileak·VDD (subthreshold + gate tunneling)
- Zero static power in ideal CMOS: never both transistors on simultaneously

NAND gate: 2 PMOS in parallel (pull-up) + 2 NMOS in series (pull-down)
NOR gate:  2 PMOS in series (pull-up) + 2 NMOS in parallel (pull-down)

Noise margin: NMH = VOH − VIH,  NML = VIL − VOL
Fanout limited by capacitive load on driving gate.
""",

    r"phonon|acoustic|optical|lattice.vibrat|debye|einstein": """
**Phonons & Lattice Dynamics** [Wikipedia/Phonon]

Quantized lattice vibration. Two branches in diatomic chain (masses M₁, M₂):

Acoustic: ω = √(2C/M)·|sin(qa/2)|  (→ 0 as q → 0, sound waves)
Optical:  ω = √(2C(M₁+M₂)/(M₁M₂))·|cos(qa/2)|  (finite ω at q=0, IR active)

Debye model: ω = vs·q (linear), ΘD = ħωD/kB (Si: ΘD ≈ 640K)
Einstein model: all modes at single ω₀ — better for optical branch

Electron-phonon coupling: primary scattering at high T → μ ∝ T^(−3/2)
Umklapp processes: phonon momentum can transfer G (reciprocal vector) → thermal resistance

Piezoelectric scattering: important in GaAs, not in Si (centrosymmetric).
""",

    r"density.of.states|dos|d\.o\.s|van hove|parabolic.band": """
**Density of States** [Wikipedia/Density of states]

3D parabolic band:  g(E) = (1/2π²)·(2m*/ħ²)^(3/2)·√(E−Ec)  [states/J/m³]

Dimensionality effects:
- 3D bulk:       g(E) ∝ √E          (smooth, parabolic)
- 2D quantum well: g(E) = m*/(πħ²)  (step function per subband)
- 1D nanowire:   g(E) ∝ 1/√(E−En)  (van Hove singularity at band edge)
- 0D quantum dot: δ-function peaks  (atomic-like discrete levels)

Occupied DoS = g(E)·f(E) → integrating gives carrier concentration:
n = ∫[Ec to ∞] g(E)·f(E) dE ≈ Nc·exp(−(Ec−Ef)/kT)  (non-degenerate approx.)

Joint DoS governs optical absorption and emission rates.
""",
}


def fallback_answer(query: str, context: str) -> str:
    """Pattern-match fallback with formula-rich responses + any retrieved context."""
    q = query.lower()
    for pattern, answer in FALLBACK_DB.items():
        if re.search(pattern, q):
            if context:
                snippet = ' '.join(context.split()[:120])
                return answer.strip() + f"\n\n*Live context retrieved: ...{snippet}...*"
            return answer.strip()

    # Generic with context
    if context:
        snippet = ' '.join(context.split()[:200])
        return (f"Based on live-retrieved sources:\n\n{snippet}\n\n"
                f"*(Install Ollama + llama3 for full AI-generated answers. "
                f"Run: `ollama pull llama3` then `ollama serve`)*")

    return ("I'm **PhysBot** — your semiconductor physics AI assistant.\n\n"
            "I can answer questions on:\n"
            "• Fermi-Dirac statistics & carrier concentrations\n"
            "• Band theory, Kronig-Penney, Brillouin zones (1st–10th)\n"
            "• Transport: Drude, mobility, Hall effect, Matthiessen's rule\n"
            "• p-n junctions, MOSFETs, CMOS logic\n"
            "• Phonons, density of states, heterostructures\n"
            "• Si, Ge, GaAs, GaN parameters\n\n"
            "**For full AI answers**: install [Ollama](https://ollama.ai) and run "
            "`ollama pull llama3`. I retrieve live knowledge from Wikipedia & ArXiv automatically.")


# ══════════════════════════════════════════════════════════════
#  MAIN RAG PIPELINE
# ══════════════════════════════════════════════════════════════

def rag_pipeline(query: str, history: list) -> dict:
    """
    Returns dict: {reply, sources, retrieved}
    """
    # 1. Retrieve
    context = retrieve(query)

    # 2. Build prompt
    messages = build_prompt(query, context, history)

    # 3. Try Ollama
    try:
        reply = query_ollama(messages)
        source_note = "llama3 via Ollama + live Wikipedia/ArXiv"
    except Exception:
        # 4. Fallback
        reply = fallback_answer(query, context)
        source_note = "rule-based fallback + live Wikipedia/ArXiv"

    # Identify cited sources
    topics = identify_topics(query)
    sources = [f"Wikipedia/{t}" for t in topics[:3]]
    if "arxiv" in context.lower() or "arxiv" in reply.lower():
        sources.append("ArXiv.org")

    return {
        "reply":   reply,
        "sources": sources,
        "engine":  source_note,
        "context_words": len(context.split()) if context else 0,
    }


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@chatbot_bp.route('/api/chat', methods=['POST'])
def chat():
    data    = request.get_json() or {}
    message = data.get('message', '').strip()
    history = data.get('history', [])
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    try:
        result = rag_pipeline(message, history)
        return jsonify(result)
    except Exception as e:
        return jsonify({'reply': f'Internal error: {str(e)}', 'sources': []}), 500


@chatbot_bp.route('/api/chat/sources', methods=['GET'])
def sources():
    """Return list of all source URLs this bot can draw from."""
    all_urls = []
    for topic, urls in TOPIC_SOURCES.items():
        for url in urls:
            all_urls.append({"topic": topic, "url": url})
    return jsonify({"sources": all_urls, "count": len(all_urls)})


@chatbot_bp.route('/api/chat/warmup', methods=['POST'])
def warmup():
    """Pre-fetch and cache all sources in background."""
    def _warm():
        for topic, urls in TOPIC_SOURCES.items():
            for url in urls[:1]:
                fetch_wikipedia(url)
    t = threading.Thread(target=_warm, daemon=True)
    t.start()
    return jsonify({"status": "warming up", "sources": len(TOPIC_SOURCES)})
