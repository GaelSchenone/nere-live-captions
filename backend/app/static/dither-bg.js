// Fondo con dithering ordenado (Bayer 4x4) sobre un degrade radial, para imitar
// la estetica "1-bit desktop" pero calculado en vivo con canvas (no es una imagen).
(function () {
  const BAYER4 = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
  ];
  const CELL = 3; // tamano de cada "pixel" del dither

  function hexToRgb(hex) {
    const n = parseInt(hex.replace('#', ''), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  const PINK = hexToRgb('#eea7c4');
  const BLACK = hexToRgb('#0c0c0c');

  function makeCanvas() {
    let canvas = document.getElementById('dither-bg');
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.id = 'dither-bg';
      document.body.prepend(canvas);
    }
    return canvas;
  }

  function render() {
    const canvas = makeCanvas();
    const ctx = canvas.getContext('2d');
    const w = window.innerWidth;
    const h = window.innerHeight;
    const cols = Math.ceil(w / CELL);
    const rows = Math.ceil(h / CELL);
    canvas.width = cols;
    canvas.height = rows;

    const img = ctx.createImageData(cols, rows);

    // un par de "manchas" radiales suaves, como en la referencia (mas densidad
    // abajo a la izquierda y a la derecha, mas clara arriba al centro)
    const blobs = [
      { x: cols * 0.18, y: rows * 0.85, r: cols * 0.55 },
      { x: cols * 0.9, y: rows * 0.5, r: cols * 0.45 },
      { x: cols * 0.5, y: rows * 0.15, r: cols * 0.3, invert: true },
    ];

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        let intensity = 0.16; // densidad base de puntos
        for (const b of blobs) {
          const d = Math.hypot(x - b.x, y - b.y) / b.r;
          const v = Math.max(0, 1 - d);
          intensity += b.invert ? -v * 0.45 : v * 0.55;
        }
        intensity = Math.max(0, Math.min(1, intensity));

        const threshold = (BAYER4[y % 4][x % 4] + 0.5) / 16;
        const isDot = intensity > threshold;
        const [r, g, b] = isDot ? BLACK : PINK;

        const idx = (y * cols + x) * 4;
        img.data[idx] = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(render, 150);
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', render);
  } else {
    render();
  }
})();
