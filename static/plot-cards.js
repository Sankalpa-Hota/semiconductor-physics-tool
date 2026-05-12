/**
 * Per-chart Render this / Close this toggles; keeps #open-plots-field in sync for form POST.
 */
(function () {
  function ph() {
    return window.PLOT_PLACEHOLDER_HTML || '';
  }

  function execScripts(container) {
    container.querySelectorAll('script').forEach(function (orig) {
      var ns = document.createElement('script');
      ns.textContent = orig.textContent;
      document.body.appendChild(ns);
      ns.remove();
    });
  }

  function setOpenPlotsField(keys) {
    var field = document.getElementById('open-plots-field');
    if (!field) return;
    var uniq = [];
    keys.forEach(function (k) {
      if (k && uniq.indexOf(k) === -1) uniq.push(k);
    });
    field.value = uniq.join(',');
  }

  function getOpenPlotsKeys() {
    var field = document.getElementById('open-plots-field');
    if (!field || !field.value.trim()) return [];
    return field.value.split(',').map(function (s) { return s.trim(); }).filter(Boolean);
  }

  function pullKey(btn) {
    var card = btn.closest('.plot-card[data-plot-key]');
    return card ? card.getAttribute('data-plot-key') : null;
  }

  function readRecipPayload() {
    var nEl = document.getElementById('recip-n');
    var n = nEl ? parseInt(nEl.value, 10) || 4 : 4;
    n = Math.max(3, Math.min(8, n));
    var verts = [];
    for (var i = 0; i < n; i++) {
      var xf = document.querySelector('[name="recip_' + i + '_x"]');
      var yf = document.querySelector('[name="recip_' + i + '_y"]');
      var zf = document.querySelector('[name="recip_' + i + '_z"]');
      if (!xf || !yf || !zf) continue;
      verts.push([
        parseFloat(xf.value) || 0,
        parseFloat(yf.value) || 0,
        parseFloat(zf.value) || 0
      ]);
    }
    return verts;
  }

  function readFormPayload() {
    if (typeof syncBzLayersHidden === 'function') syncBzLayersHidden();
    var form = document.getElementById('sim-form');
    var o = {};
    if (!form) return o;
    var fd = new FormData(form);
    fd.forEach(function (v, k) {
      if (k === 'open_plots') return;
      o[k] = v;
    });
    if (typeof bzState !== 'undefined' && bzState) {
      o.bz_lattice = bzState.lattice;
      o.bz_a = String(bzState.a);
      o.bz_b = String(bzState.b);
      o.bz_angle = String(bzState.angle);
      o.bz_zones = String(bzState.zones);
      o.bz_layers = (typeof getBzLayersPayload === 'function')
        ? getBzLayersPayload()
        : (document.getElementById('hid-bz-layers') || {}).value || 'all';
    }
    var rv = readRecipPayload();
    if (rv.length >= 3) o.recip_vertices = rv;
    var rca = document.querySelector('[name="recip_c_axis"]');
    if (rca && rca.value !== '') o.recip_c_axis = rca.value;
    var rn = document.getElementById('recip-n');
    if (rn && rn.value) o.recip_n = rn.value;
    var as = document.getElementById('active-section-field');
    if (as && as.value) o.active_section = as.value;
    return o;
  }

  function markPlotKeyOpen(key) {
    if (!key) return;
    var card = document.querySelector('.plot-card[data-plot-key="' + key + '"]');
    if (!card) return;
    var btn = card.querySelector('.plot-toggle-btn');
    if (btn) {
      btn.textContent = 'Close this';
      btn.setAttribute('data-state', 'open');
      btn.setAttribute('aria-expanded', 'true');
    }
    var cur = getOpenPlotsKeys().filter(function (k) { return k !== key; });
    cur.push(key);
    setOpenPlotsField(cur);
  }

  window.markPlotKeyOpen = markPlotKeyOpen;

  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.plot-card[data-plot-key] .plot-toggle-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var key = pullKey(btn);
        if (!key) return;
        var card = btn.closest('.plot-card[data-plot-key]');
        var inner = card ? card.querySelector('.plot-inner') : null;
        if (!inner) return;

        var open = btn.getAttribute('data-state') === 'open';
        if (open) {
          inner.innerHTML = ph();
          if (key === 'bz_plot') {
            inner.style.minHeight = '';
          }
          btn.textContent = 'Render this';
          btn.setAttribute('data-state', 'closed');
          btn.setAttribute('aria-expanded', 'false');
          setOpenPlotsField(getOpenPlotsKeys().filter(function (k) { return k !== key; }));
          return;
        }

        btn.textContent = 'Loading…';
        btn.disabled = true;
        var payload = readFormPayload();
        payload.plot = key;

        fetch('/api/plot', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        })
          .then(function (r) { return r.json(); })
          .then(function (data) {
            if (!Object.prototype.hasOwnProperty.call(data, 'plot')) {
              throw new Error(data.error || 'Unknown error');
            }
            inner.innerHTML = data.plot;
            execScripts(inner);
            if (key === 'bz_plot') {
              inner.style.minHeight = '440px';
            }
            btn.textContent = 'Close this';
            btn.setAttribute('data-state', 'open');
            btn.setAttribute('aria-expanded', 'true');
            var cur = getOpenPlotsKeys().filter(function (k) { return k !== key; });
            cur.push(key);
            setOpenPlotsField(cur);
          })
          .catch(function (err) {
            console.error(err);
            inner.innerHTML =
              '<div class="plot-error">Could not render this plot. Adjust parameters and try again, or click <strong>Compute parameters</strong>.</div>';
            btn.textContent = 'Render this';
            btn.setAttribute('data-state', 'closed');
            btn.setAttribute('aria-expanded', 'false');
          })
          .finally(function () {
            btn.disabled = false;
          });
      });
    });
  });
})();
