import { initGPU, loadScan, loadSort } from '../kit/src/index.js';
import { createVisualiser } from './visualiser.js';
import { createBenchmark } from './benchmark.js';

async function start() {
  const { device } = await initGPU({ features: ['timestamp-query'] });
  const sort = await loadSort(device);
  const scan = await loadScan(device);
  const vis = createVisualiser(device, sort);
  createBenchmark(device, sort, scan);
  await vis.show('bitonic', true);
}

start().catch(e => {
  console.warn(e);
  document.body.classList.add('no-gpu');
  // the note in the page says there's no WebGPU. anything else gets its own message
  if (navigator.gpu) document.querySelector('.nogpu').textContent = `Couldn't start: ${e.message}`;
  document.getElementById('vis-label').textContent = '';
  for (const id of ['bench-run', 'vis-play', 'vis-radix', 'vis-bitonic', 'vis-file'])
    document.getElementById(id).disabled = true;
});
