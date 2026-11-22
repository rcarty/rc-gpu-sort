// sorts the shuffled pixels of a picture and reads the array back after every stage
import { buf, readBack } from '../kit/src/index.js';
import { randomKeys } from './keys.js';

const SIDE = 512;
const VIS_N = SIDE * SIDE;
const SHIFT = 14; // top 18 bits of a key are the pixel's home, the rest is random
const ALGOS = ['bitonic', 'radix'];
const MOVE_MS = 900;
const HOLD_MS = 350;
const BG = 0xff0e0b0a; // abgr
const $ = id => document.getElementById(id);

// default picture: sun going down behind some hills
function sunset(g) {
  const S = SIDE;
  const horizon = S * 0.62;
  const sky = g.createLinearGradient(0, 0, 0, horizon);
  sky.addColorStop(0, '#0d1430');
  sky.addColorStop(0.55, '#3b4f8f');
  sky.addColorStop(0.85, '#e0825a');
  sky.addColorStop(1, '#ffc27a');
  g.fillStyle = sky;
  g.fillRect(0, 0, S, horizon);
  g.fillStyle = '#fff1c4';
  g.beginPath();
  g.arc(S * 0.63, horizon - S * 0.1, S * 0.13, 0, 7);
  g.fill();
  const hills = [
    ['#4a5a94', 0.14, 0.011, 1],
    ['#2c3a6c', 0.09, 0.019, 3],
    ['#141c3c', 0.05, 0.031, 5],
  ];
  for (const [fill, height, freq, phase] of hills) {
    g.fillStyle = fill;
    g.beginPath();
    g.moveTo(0, horizon);
    for (let x = 0; x <= S; x += 4) {
      const h = 0.6 + 0.3 * Math.sin(x * freq + phase) + 0.1 * Math.sin(x * freq * 3.7 + phase);
      g.lineTo(x, horizon - S * height * h);
    }
    g.lineTo(S, horizon);
    g.fill();
  }
  const sea = g.createLinearGradient(0, horizon, 0, S);
  sea.addColorStop(0, '#b8694f');
  sea.addColorStop(0.25, '#2c3a6c');
  sea.addColorStop(1, '#0a0f24');
  g.fillStyle = sea;
  g.fillRect(0, horizon, S, S - horizon);
  g.fillStyle = 'rgba(255, 226, 170, 0.5)';
  for (let y = horizon + 6; y < S; y += 9) {
    const w = S * 0.05 + (y - horizon) * 0.5 * (0.6 + 0.4 * Math.sin(y * 1.7));
    g.fillRect(S * 0.63 - w / 2, y, w, 2);
  }
}

function describe(algo, step, steps) {
  if (step === 0) return ['The picture', '262,144 pixels, each keyed by where it belongs'];
  if (step === 1) return ['Shuffled', 'Random order. This is the input to the sort'];
  const s = step - 1;
  const n = steps - 1;
  if (algo !== 'bitonic') {
    const done = step === steps ? '. Done.' : '';
    const what =
      s < 4 ? "Low digits. These don't show yet" : `Sorted on the low ${s * 4} of 32 bits${done}`;
    return [`Pass ${s} of ${n}`, what];
  }
  const side = 1 << (s - 1); // a sorted run of 512 * side keys draws as copies this wide
  let what = `Runs of ${(512 * side).toLocaleString()} keys sorted`;
  if (side >= 32) what = `${(SIDE / side) ** 2} rough copies of the picture, ${side} pixels wide`;
  if (s === n) what = 'One picture. Done.';
  return [`Stage ${s} of ${n}`, what];
}

export function createVisualiser(device, sort) {
  const plot = $('plot');
  const g2 = plot.getContext('2d');
  const img = g2.createImageData(SIDE, SIDE);
  const out = new Uint32Array(img.data.buffer);
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;

  const pic = document.createElement('canvas');
  pic.width = pic.height = SIDE;
  const pg = pic.getContext('2d', { willReadFrequently: true });
  sunset(pg);
  let colour; // abgr per pixel, row by row

  const vis = {
    algo: 'bitonic',
    snaps: {},
    dests: {},
    pos: 0, // step, with a fraction while pixels are moving
    playing: false,
  };

  function readPicture() {
    colour = new Uint32Array(pg.getImageData(0, 0, SIDE, SIDE).data.buffer);
    for (let i = 0; i < VIS_N; i++) colour[i] |= 0xff000000;
  }

  // the picture in order, then shuffled, then after every sort stage
  async function snapshots(algo) {
    const U = GPUBufferUsage;
    const rnd = randomKeys(VIS_N, 12345);
    const sorted = new Uint32Array(VIS_N);
    for (let i = 0; i < VIS_N; i++) sorted[i] = ((i << SHIFT) | (rnd[i] >>> (32 - SHIFT))) >>> 0;
    const input = sorted.slice();
    for (let i = VIS_N - 1; i > 0; i--) {
      const j = rnd[i] % (i + 1);
      const t = input[i];
      input[i] = input[j];
      input[j] = t;
    }
    const keys = buf(device, VIS_N * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST);
    const tmpKeys = buf(device, VIS_N * 4, U.STORAGE | U.COPY_SRC);
    device.queue.writeBuffer(keys, 0, input);
    const sorter = sort[algo]({ count: VIS_N, keys, tmpKeys });
    const snaps = [sorted, input];
    try {
      for (let s = 0; s < sorter.steps; s++) {
        const e = device.createCommandEncoder();
        const cp = e.beginComputePass();
        sorter.encodeStep(cp, s);
        cp.end();
        device.queue.submit([e.finish()]);
        snaps.push(await readBack(device, sorter.outputAfterStep(s).keys, VIS_N * 4));
      }
    } finally {
      sorter.destroy();
      keys.destroy();
      tmpKeys.destroy();
    }
    return snaps;
  }

  // where each pixel in snapshot a sits in snapshot b
  function moves(a, b) {
    const where = new Uint32Array(VIS_N);
    for (let j = 0; j < VIS_N; j++) where[b[j] >>> SHIFT] = j;
    return a.map(k => where[k >>> SHIFT]);
  }

  function draw() {
    const snaps = vis.snaps[vis.algo];
    const steps = snaps ? snaps.length - 1 : 0;
    if (snaps) {
      const s = Math.min(Math.floor(vis.pos), steps);
      const t = vis.pos - s;
      const a = snaps[s];
      if (t > 0) {
        const ease = t * t * (3 - 2 * t);
        const dest = vis.dests[vis.algo][s];
        out.fill(BG);
        for (let i = 0; i < VIS_N; i++) {
          const d = dest[i];
          const x0 = i % SIDE;
          const y0 = (i / SIDE) | 0;
          const x = (x0 + ((d % SIDE) - x0) * ease + 0.5) | 0;
          const y = (y0 + (((d / SIDE) | 0) - y0) * ease + 0.5) | 0;
          out[y * SIDE + x] = colour[a[i] >>> SHIFT];
        }
      } else {
        for (let i = 0; i < VIS_N; i++) out[i] = colour[a[i] >>> SHIFT];
      }
    } else {
      out.fill(BG);
    }
    g2.putImageData(img, 0, 0);

    $('vis-step').max = steps || 1;
    const [label, desc] = describe(vis.algo, Math.ceil(vis.pos - 1e-6), steps);
    $('vis-label').textContent = snaps ? label : 'Sorting on the GPU…';
    $('vis-desc').textContent = snaps ? desc : '';
  }

  function setPos(pos) {
    vis.pos = Math.max(0, pos);
    draw();
    $('vis-step').value = vis.pos;
  }

  let raf = 0;
  function stop() {
    vis.playing = false;
    cancelAnimationFrame(raf); // or a restart in the same task leaves two loops running
    $('vis-play').textContent = 'Play';
  }

  function play() {
    const snaps = vis.snaps[vis.algo];
    if (!snaps || vis.playing) return;
    const steps = snaps.length - 1;
    if (vis.pos >= steps) setPos(0);
    vis.playing = true;
    $('vis-play').textContent = 'Pause';
    let from = Math.floor(vis.pos);
    let ms = (vis.pos - from) * MOVE_MS; // time into the current step
    let last = 0;
    const tick = now => {
      if (!vis.playing) return;
      ms += last ? Math.min(100, Math.max(0, now - last)) : 0;
      last = now;
      if (ms >= MOVE_MS + HOLD_MS) {
        from++;
        ms = 0;
      }
      if (from >= steps) {
        setPos(steps);
        return stop();
      }
      // with reduced motion the pixels jump to each step
      const t = Math.min(1, ms / MOVE_MS);
      setPos(from + (still ? Math.floor(t) : t));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
  }

  async function show(algo, autoplay = false) {
    stop();
    vis.algo = algo;
    for (const k of ALGOS) $('vis-' + k).setAttribute('aria-pressed', algo === k);
    $('vis-play').disabled = !vis.snaps[algo];
    setPos(0);
    if (!vis.snaps[algo]) {
      try {
        const snaps = await snapshots(algo);
        vis.dests[algo] = snaps.slice(1).map((next, s) => moves(snaps[s], next));
        vis.snaps[algo] = snaps;
      } catch (e) {
        console.warn(e);
        $('vis-label').textContent = 'The sort failed on this GPU';
        $('vis-desc').textContent = e.message;
        return;
      }
      if (vis.algo !== algo) return;
    }
    $('vis-play').disabled = false;
    setPos(0);
    if (autoplay && !still) play();
  }

  for (const k of ALGOS) $('vis-' + k).onclick = () => show(k, true);
  $('vis-step').oninput = e => {
    stop();
    setPos(+e.target.value);
  };
  $('vis-play').onclick = () => (vis.playing ? stop() : play());

  // sort your own picture. it's read in the page and never uploaded
  $('vis-file').onchange = async e => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const bmp = await createImageBitmap(file);
      const side = Math.min(bmp.width, bmp.height);
      const sx = (bmp.width - side) / 2;
      const sy = (bmp.height - side) / 2;
      pg.clearRect(0, 0, SIDE, SIDE); // or the last picture shows through transparent areas
      pg.drawImage(bmp, sx, sy, side, side, 0, 0, SIDE, SIDE);
      bmp.close();
    } catch {
      stop(); // draw() rewrites this line every frame while playing
      $('vis-desc').textContent = "Couldn't read that image";
      return;
    }
    readPicture();
    stop();
    setPos(0);
    if (!still) play();
  };

  readPicture();
  return { show };
}
