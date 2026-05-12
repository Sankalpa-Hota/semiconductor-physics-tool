/**
 * circuit.js — Interactive animated semiconductor circuit board
 * Renders on a <canvas id="circuit-canvas">
 * Features: PCB traces, glowing nodes, moving electron particles,
 * MOSFET/BJT/capacitor/resistor symbols, interactive hover.
 */

(function () {
  const canvas = document.getElementById('circuit-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // ── Colors ────────────────────────────────────────────────
  const C = {
    bg:       '#050810',
    trace:    '#0d2035',
    traceHi:  '#00f0ff',
    node:     '#00f0ff',
    nodeGlow: 'rgba(0,240,255,0.25)',
    electron: '#00ff9d',
    label:    'rgba(0,240,255,0.55)',
    resistor: '#ffb800',
    cap:      '#9b5de5',
    mos:      '#ff3e8a',
    grid:     'rgba(255,255,255,0.018)',
  };

  let W, H, dpr;
  const nodes   = [];
  const traces  = [];
  const symbols = [];
  const electrons = [];

  // ── Resize ─────────────────────────────────────────────────
  function resize() {
    dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    W = rect.width; H = rect.height;
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);
    build();
  }

  // ── Build circuit topology ──────────────────────────────────
  function build() {
    nodes.length = traces.length = symbols.length = electrons.length = 0;

    const cols = Math.floor(W / 90);
    const rows = Math.floor(H / 70);
    const gx = W / cols;
    const gy = H / rows;

    // Grid snap helper
    const snap = (i, j) => ({ x: gx * i + gx * 0.5, y: gy * j + gy * 0.5 });

    // Create nodes on a sparse grid
    const used = new Set();
    const key = (i, j) => `${i},${j}`;

    function addNode(i, j) {
      if (i < 0 || i >= cols || j < 0 || j >= rows) return null;
      const k = key(i, j);
      if (used.has(k)) return nodes.find(n => n.gi === i && n.gj === j);
      used.add(k);
      const p = snap(i, j);
      const node = { x: p.x, y: p.y, gi: i, gj: j,
                     r: 3.5, pulse: Math.random() * Math.PI * 2,
                     active: false };
      nodes.push(node);
      return node;
    }

    // Deterministic-ish layout seeded by grid size
    const seed = cols * rows;
    function pseudoRand(n) { return ((n * 1664525 + 1013904223) & 0xffffffff) / 0x100000000; }

    // Place nodes
    let ni = 0;
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const r = pseudoRand(ni++);
        if (r > 0.35) addNode(i, j);
      }
    }

    // Traces: connect adjacent nodes horizontally/vertically
    for (const n of nodes) {
      // right
      const nr = nodes.find(m => m.gi === n.gi + 1 && m.gj === n.gj);
      if (nr && pseudoRand(n.gi * 17 + n.gj * 31) > 0.3) {
        traces.push({ a: n, b: nr, lit: false, litT: 0 });
      }
      // down
      const nd = nodes.find(m => m.gi === n.gi && m.gj === n.gj + 1);
      if (nd && pseudoRand(n.gi * 23 + n.gj * 13) > 0.4) {
        traces.push({ a: n, b: nd, lit: false, litT: 0 });
      }
    }

    // Place symbols on some traces
    const symTypes = ['R', 'C', 'L', 'M', 'D', 'T'];
    let si = 0;
    for (const t of traces) {
      if (pseudoRand(si++ * 97 + 7) > 0.72) {
        const mx = (t.a.x + t.b.x) / 2;
        const my = (t.a.y + t.b.y) / 2;
        const horiz = Math.abs(t.b.x - t.a.x) > Math.abs(t.b.y - t.a.y);
        const type = symTypes[Math.floor(pseudoRand(si * 53) * symTypes.length)];
        symbols.push({ x: mx, y: my, type, horiz, trace: t, hover: false });
      }
    }

    // Spawn electrons
    spawnElectrons();
  }

  // ── Electrons ──────────────────────────────────────────────
  function spawnElectrons() {
    const count = Math.min(traces.length, Math.floor(W / 25));
    for (let i = 0; i < count; i++) {
      const t = traces[Math.floor(Math.random() * traces.length)];
      electrons.push({
        trace: t, t: Math.random(),
        speed: 0.003 + Math.random() * 0.006,
        dir: Math.random() > 0.5 ? 1 : -1,
        alpha: 0.7 + Math.random() * 0.3,
        size: 2 + Math.random() * 1.5,
      });
    }
  }

  // ── Mouse interaction ───────────────────────────────────────
  let mouseX = -999, mouseY = -999;
  canvas.addEventListener('mousemove', e => {
    const r = canvas.getBoundingClientRect();
    mouseX = e.clientX - r.left;
    mouseY = e.clientY - r.top;
  });
  canvas.addEventListener('mouseleave', () => { mouseX = -999; mouseY = -999; });

  // ── Draw helpers ────────────────────────────────────────────
  function drawGrid() {
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 0.5;
    const step = 30;
    for (let x = 0; x < W; x += step) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y < H; y += step) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }
  }

  function drawTrace(t, now) {
    const lit = t.litT > 0;
    const alpha = lit ? 0.9 : 0.35;
    const grd = ctx.createLinearGradient(t.a.x, t.a.y, t.b.x, t.b.y);
    if (lit) {
      grd.addColorStop(0,   `rgba(0,240,255,${alpha * 0.4})`);
      grd.addColorStop(0.5, `rgba(0,240,255,${alpha})`);
      grd.addColorStop(1,   `rgba(0,240,255,${alpha * 0.4})`);
    } else {
      grd.addColorStop(0,   `rgba(13,32,53,${alpha})`);
      grd.addColorStop(1,   `rgba(13,32,53,${alpha})`);
    }
    ctx.beginPath();
    ctx.moveTo(t.a.x, t.a.y);
    ctx.lineTo(t.b.x, t.b.y);
    ctx.strokeStyle = grd;
    ctx.lineWidth = lit ? 1.8 : 1.2;
    ctx.stroke();
    if (t.litT > 0) t.litT -= 0.018;
  }

  function drawNode(n, now) {
    const dist = Math.hypot(mouseX - n.x, mouseY - n.y);
    const hover = dist < 22;
    if (hover) { n.active = true; n.pulse = now * 0.002; }

    const pulse = 0.6 + 0.4 * Math.sin(n.pulse + now * 0.0015);
    const r = n.r * (hover ? 1.6 : 1);
    const glow = hover ? 20 : 10;

    // Glow ring
    const grad = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, glow);
    grad.addColorStop(0, `rgba(0,240,255,${0.3 * pulse})`);
    grad.addColorStop(1, 'rgba(0,240,255,0)');
    ctx.beginPath();
    ctx.arc(n.x, n.y, glow, 0, Math.PI * 2);
    ctx.fillStyle = grad; ctx.fill();

    // Core dot
    ctx.beginPath();
    ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
    ctx.fillStyle = hover ? '#ffffff' : `rgba(0,240,255,${0.7 + 0.3 * pulse})`;
    ctx.fill();

    // Hover: ripple + activate traces
    if (hover) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 5 + 4 * Math.sin(now * 0.003), 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(0,240,255,0.3)'; ctx.lineWidth = 1;
      ctx.stroke();
      // Light up connected traces
      for (const t of traces) {
        if (t.a === n || t.b === n) t.litT = 1.2;
      }
    }
  }

  function drawElectron(e, now) {
    const t = e.trace;
    const tVal = e.t;
    const x = t.a.x + (t.b.x - t.a.x) * tVal;
    const y = t.a.y + (t.b.y - t.a.y) * tVal;

    const grd = ctx.createRadialGradient(x, y, 0, x, y, e.size * 3);
    grd.addColorStop(0, `rgba(0,255,157,${e.alpha})`);
    grd.addColorStop(1, 'rgba(0,255,157,0)');
    ctx.beginPath();
    ctx.arc(x, y, e.size * 2.5, 0, Math.PI * 2);
    ctx.fillStyle = grd; ctx.fill();

    ctx.beginPath();
    ctx.arc(x, y, e.size * 0.7, 0, Math.PI * 2);
    ctx.fillStyle = '#afffdf'; ctx.fill();

    // Advance
    e.t += e.speed * e.dir;
    if (e.t > 1 || e.t < 0) {
      e.dir *= -1;
      e.t = Math.max(0, Math.min(1, e.t));
      // Light up trace
      t.litT = 1.0;
    }
  }

  // ── Component symbols ───────────────────────────────────────
  function drawSymbol(s, now) {
    const dist = Math.hypot(mouseX - s.x, mouseY - s.y);
    s.hover = dist < 18;
    const alpha = s.hover ? 1 : 0.7;
    ctx.save();
    ctx.translate(s.x, s.y);
    if (!s.horiz) ctx.rotate(Math.PI / 2);

    switch (s.type) {
      case 'R': drawResistor(alpha, s.hover, now); break;
      case 'C': drawCapacitor(alpha, s.hover, now); break;
      case 'L': drawInductor(alpha, s.hover, now); break;
      case 'M': drawMOSFET(alpha, s.hover, now); break;
      case 'D': drawDiode(alpha, s.hover, now); break;
      case 'T': drawBJT(alpha, s.hover, now); break;
    }

    if (s.hover) {
      ctx.fillStyle = 'rgba(0,240,255,0.85)';
      ctx.font = '600 9px Space Mono, monospace';
      ctx.textAlign = 'center';
      const labels = { R:'Resistor', C:'Capacitor', L:'Inductor',
                       M:'MOSFET', D:'Diode p-n', T:'BJT' };
      ctx.fillText(labels[s.type], 0, -16);
    }
    ctx.restore();
  }

  function drawResistor(alpha, hover, now) {
    const col = hover
      ? `rgba(255,184,0,1)`
      : `rgba(255,184,0,${alpha})`;
    ctx.strokeStyle = col; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(-12, 0); ctx.lineTo(-7, 0); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(7, 0); ctx.lineTo(12, 0); ctx.stroke();
    ctx.strokeRect(-7, -4, 14, 8);
    if (hover) {
      ctx.shadowColor = '#ffb800'; ctx.shadowBlur = 8;
      ctx.strokeRect(-7, -4, 14, 8);
      ctx.shadowBlur = 0;
    }
  }

  function drawCapacitor(alpha, hover, now) {
    const col = hover ? '#9b5de5' : `rgba(155,93,229,${alpha})`;
    ctx.strokeStyle = col; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(-12, 0); ctx.lineTo(-4, 0); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(4, 0); ctx.lineTo(12, 0); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(-4, -7); ctx.lineTo(-4, 7); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(4, -7); ctx.lineTo(4, 7); ctx.stroke();
    if (hover) {
      ctx.shadowColor = '#9b5de5'; ctx.shadowBlur = 10;
      ctx.beginPath(); ctx.moveTo(-4, -7); ctx.lineTo(-4, 7); ctx.stroke();
      ctx.shadowBlur = 0;
    }
  }

  function drawInductor(alpha, hover, now) {
    const col = hover ? '#00f0ff' : `rgba(0,240,255,${alpha})`;
    ctx.strokeStyle = col; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(-12, 0); ctx.lineTo(-9, 0); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(9, 0); ctx.lineTo(12, 0); ctx.stroke();
    // 3 bumps
    for (let i = 0; i < 3; i++) {
      ctx.beginPath();
      ctx.arc(-6 + i * 6, 0, 3, Math.PI, 0);
      ctx.stroke();
    }
  }

  function drawMOSFET(alpha, hover, now) {
    const col = hover ? '#ff3e8a' : `rgba(255,62,138,${alpha})`;
    ctx.strokeStyle = col; ctx.lineWidth = 1.5;
    // Gate line
    ctx.beginPath(); ctx.moveTo(-12, 0); ctx.lineTo(-5, 0); ctx.stroke();
    // Gate bar
    ctx.beginPath(); ctx.moveTo(-5, -8); ctx.lineTo(-5, 8); ctx.stroke();
    // Channel
    ctx.beginPath(); ctx.moveTo(-3, -8); ctx.lineTo(-3, 8); ctx.stroke();
    // Drain / Source
    ctx.beginPath(); ctx.moveTo(-3, -5); ctx.lineTo(12, -5); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(-3, 5); ctx.lineTo(12, 5); ctx.stroke();
    // Arrow
    ctx.beginPath();
    ctx.moveTo(2, 5); ctx.lineTo(-1, 0); ctx.lineTo(2, -5);
    ctx.strokeStyle = col; ctx.stroke();
    if (hover) { ctx.shadowColor = '#ff3e8a'; ctx.shadowBlur = 10; ctx.stroke(); ctx.shadowBlur = 0; }
  }

  function drawDiode(alpha, hover, now) {
    const col = hover ? '#00ff9d' : `rgba(0,255,157,${alpha})`;
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(-12, 0); ctx.lineTo(-5, 0); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(5, 0); ctx.lineTo(12, 0); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(-5, -6); ctx.lineTo(-5, 6); ctx.lineTo(5, 0); ctx.closePath();
    ctx.globalAlpha = hover ? 0.4 : 0.15;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.stroke();
    ctx.beginPath(); ctx.moveTo(5, -6); ctx.lineTo(5, 6); ctx.stroke();
  }

  function drawBJT(alpha, hover, now) {
    const col = hover ? '#4cc9f0' : `rgba(76,201,240,${alpha})`;
    ctx.strokeStyle = col; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(0, 0, 9, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(-12, 0); ctx.lineTo(-9, 0); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(-9, -7); ctx.lineTo(-9, 7); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(-9, -4); ctx.lineTo(4, -8); ctx.lineTo(12, -10); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(-9, 4); ctx.lineTo(4, 8); ctx.lineTo(12, 10); ctx.stroke();
    // Arrow on emitter
    ctx.beginPath();
    ctx.moveTo(6, 7); ctx.lineTo(4, 10); ctx.lineTo(1, 7);
    ctx.fillStyle = col; ctx.fill();
  }

  // ── Main loop ───────────────────────────────────────────────
  let rafId;
  function frame(now) {
    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);

    // Subtle radial gradient
    const radGrd = ctx.createRadialGradient(W*0.5, H*0.5, 0, W*0.5, H*0.5, W*0.6);
    radGrd.addColorStop(0, 'rgba(0,240,255,0.03)');
    radGrd.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = radGrd; ctx.fillRect(0, 0, W, H);

    drawGrid();

    for (const t of traces)  drawTrace(t, now);
    for (const e of electrons) drawElectron(e, now);
    for (const s of symbols)  drawSymbol(s, now);
    for (const n of nodes)   drawNode(n, now);

    rafId = requestAnimationFrame(frame);
  }

  // ── Init ────────────────────────────────────────────────────
  resize();
  window.addEventListener('resize', () => {
    cancelAnimationFrame(rafId);
    resize();
    requestAnimationFrame(frame);
  });
  requestAnimationFrame(frame);
})();
