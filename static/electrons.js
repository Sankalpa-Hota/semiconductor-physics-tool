/**
 * Five decorative “electrons” drift across the viewport with randomizing directions.
 */
(function () {
  var field = document.getElementById('electron-field');
  if (!field) return;
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    field.remove();
    return;
  }

  var COUNT = 5;
  var SIZE = 36;
  var PAD = 8;
  var MIN_SPD = 30;
  var MAX_SPD = 100;

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function bounds() {
    var w = window.innerWidth;
    var h = window.innerHeight;
    return {
      maxX: Math.max(PAD, w - SIZE - PAD),
      maxY: Math.max(PAD, h - SIZE - PAD)
    };
  }

  var b0 = bounds();
  var particles = [];
  for (var i = 0; i < COUNT; i++) {
    var el = document.createElement('div');
    el.className = 'floating-electron';
    el.setAttribute('aria-hidden', 'true');
    el.textContent = 'e-';
    field.appendChild(el);
    particles.push({
      el: el,
      px: PAD + Math.random() * (b0.maxX - PAD),
      py: PAD + Math.random() * (b0.maxY - PAD),
      angle: Math.random() * Math.PI * 2,
      speed: MIN_SPD + Math.random() * (MAX_SPD - MIN_SPD)
    });
  }

  var last = performance.now();

  function tick(now) {
    var dt = Math.min(0.045, (now - last) / 1000);
    last = now;

    var maxX = bounds().maxX;
    var maxY = bounds().maxY;

    for (var j = 0; j < particles.length; j++) {
      var p = particles[j];
      p.angle += (Math.random() - 0.5) * 3.4 * dt;
      p.speed += (Math.random() - 0.5) * 45 * dt;
      p.speed = clamp(p.speed, MIN_SPD, MAX_SPD);

      p.px += Math.cos(p.angle) * p.speed * dt;
      p.py += Math.sin(p.angle) * p.speed * dt;

      if (p.px <= PAD) {
        p.px = PAD;
        p.angle = Math.PI - p.angle + (Math.random() - 0.5) * 0.9;
      } else if (p.px >= maxX) {
        p.px = maxX;
        p.angle = Math.PI - p.angle + (Math.random() - 0.5) * 0.9;
      }
      if (p.py <= PAD) {
        p.py = PAD;
        p.angle = -p.angle + (Math.random() - 0.5) * 0.9;
      } else if (p.py >= maxY) {
        p.py = maxY;
        p.angle = -p.angle + (Math.random() - 0.5) * 0.9;
      }

      p.px = clamp(p.px, PAD, maxX);
      p.py = clamp(p.py, PAD, maxY);

      p.el.style.transform =
        'translate3d(' + p.px.toFixed(1) + 'px,' + p.py.toFixed(1) + 'px,0)';
    }
    requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);

  window.addEventListener(
    'resize',
    function () {
      var maxX = bounds().maxX;
      var maxY = bounds().maxY;
      for (var k = 0; k < particles.length; k++) {
        particles[k].px = clamp(particles[k].px, PAD, maxX);
        particles[k].py = clamp(particles[k].py, PAD, maxY);
      }
    },
    { passive: true }
  );
})();
