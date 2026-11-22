import { buf, readBack, medianGpuTime } from '../kit/src/index.js';
import { randomKeys, equal } from './keys.js';

const RUNS = 5;
const $ = id => document.getElementById(id);
const fmtMs = ms =>
  ms < 10 ? ms.toFixed(2) + ' ms' : ms < 100 ? ms.toFixed(1) + ' ms' : Math.round(ms) + ' ms';
const median = times => times.sort((a, b) => a - b)[times.length >> 1];

function render(body, rows, n, unit) {
  const max = Math.max(...rows.filter(r => r.ms).map(r => r.ms));
  body.innerHTML = '';
  for (const r of rows) {
    const tr = document.createElement('tr');
    const td = (txt, cls) => {
      const c = document.createElement('td');
      if (cls) c.className = cls;
      c.textContent = txt;
      tr.appendChild(c);
      return c;
    };
    td(r.name);
    td(r.ms ? fmtMs(r.ms) : '–', 'num');
    td(r.wall ? fmtMs(r.wall) : '–', 'num');
    const m = r.ms ? n / r.ms / 1000 : 0;
    td(r.ms ? `${m < 10 ? m.toFixed(1) : m.toFixed(0)} ${unit}` : '–', 'num');
    const bc = td('');
    if (r.ms) {
      const b = document.createElement('div');
      b.className = 'bar' + (r.cpu ? ' cpu' : '');
      b.style.width = ((100 * r.ms) / max).toFixed(1) + '%';
      bc.appendChild(b);
    }
    const v = td(r.ok === undefined ? '' : r.ok ? 'Matches CPU' : 'Mismatch');
    if (r.ok !== undefined) v.className = r.ok ? 'ok' : 'bad';
    body.appendChild(tr);
  }
}

export function createBenchmark(device, sort, scan) {
  async function run() {
    const n = +$('bench-n').value;
    const status = t => {
      $('bench-status').textContent = t;
    };
    $('bench-run').disabled = true;
    const owned = []; // freed in finally, so a failed run doesn't hold 200 MB of buffers
    const own = x => (owned.push(x), x);
    try {
      const U = GPUBufferUsage;
      const input = randomKeys(n, 987654321);
      const src = own(buf(device, n * 4, U.COPY_SRC | U.COPY_DST));
      const keys = own(buf(device, n * 4, U.STORAGE | U.COPY_SRC | U.COPY_DST));
      const tmpKeys = own(buf(device, n * 4, U.STORAGE));
      device.queue.writeBuffer(src, 0, input);
      const fresh = e => e.copyBufferToBuffer(src, 0, keys, 0, n * 4); // same input every run
      let timestamps = false;

      const sorts = [
        { name: 'Radix, WebGPU', algo: 'radix', label: 'Radix sort' },
        { name: 'Bitonic, WebGPU', algo: 'bitonic', label: 'Bitonic sort' },
        { name: 'TypedArray.sort, CPU', cpu: true },
      ];
      const scans = [{ name: 'Prefix sum, WebGPU' }, { name: 'Loop, CPU', cpu: true }];
      const show = () => {
        render($('bench-body'), sorts, n, 'Mkeys/s');
        render($('scan-body'), scans, n, 'M/s');
      };
      show();

      status('CPU sort…');
      await new Promise(r => setTimeout(r, 30));
      const cpuTimes = [];
      let sorted = null;
      for (let r = 0; r < (n > 4e6 ? 1 : 3); r++) {
        sorted = input.slice();
        const t0 = performance.now();
        sorted.sort();
        cpuTimes.push(performance.now() - t0);
      }
      sorts.at(-1).ms = median(cpuTimes);
      show();

      for (const row of sorts.filter(r => r.algo)) {
        status(`${row.label} on the GPU…`);
        const sorter = own(sort[row.algo]({ count: n, keys, tmpKeys }));
        const t = await medianGpuTime(device, {
          runs: RUNS,
          prepare: fresh,
          encode: cp => sorter.encode(cp),
        });
        row.ms = t.ms;
        row.wall = t.wallMs;
        timestamps = t.timestamps;
        row.ok = equal(await readBack(device, keys, n * 4), sorted);
        show();
      }

      // prefix sum over the same values, wrapping at 32 bits like the GPU does
      status('Prefix sum…');
      const summed = new Uint32Array(n);
      const loopTimes = [];
      for (let r = 0; r < 3; r++) {
        const t0 = performance.now();
        let sum = 0;
        for (let i = 0; i < n; i++) {
          summed[i] = sum;
          sum = (sum + input[i]) >>> 0;
        }
        loopTimes.push(performance.now() - t0);
      }
      scans[1].ms = median(loopTimes);
      const scanner = own(scan.exclusive({ count: n, data: keys }));
      const t = await medianGpuTime(device, {
        runs: RUNS,
        prepare: fresh,
        encode: cp => scanner.encode(cp),
      });
      scans[0].ms = t.ms;
      scans[0].wall = t.wallMs;
      scans[0].ok = equal(await readBack(device, keys, n * 4), summed);
      show();
      status(
        `${n.toLocaleString()} keys · median of ${RUNS} GPU runs · ${timestamps ? 'GPU timestamp queries' : 'wall clock (timestamp queries not available)'}`,
      );
    } catch (e) {
      console.warn(e);
      status('Failed: ' + e.message);
    } finally {
      for (const o of owned) o.destroy();
      $('bench-run').disabled = false;
    }
  }
  $('bench-run').onclick = run;
}
