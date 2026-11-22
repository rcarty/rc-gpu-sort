# GPU sorting: WebGPU and CUDA

Two parallel sorts for 32-bit keys and the prefix sum under them, written once in WGSL for WebGPU and ported to CUDA with the same structure, so the two APIs can be compared on the same hardware.

- **Prefix sum.** An exclusive scan over 32-bit values, and the piece the radix sort is built on. Each 256-element block is scanned in workgroup memory (Hillis-Steele), block totals are scanned one level up, and a second sweep adds them back down. Three levels cover 16 million values.
- **LSD radix sort.** Four bits per pass, eight passes. Each pass builds a per-tile digit histogram (1,024 keys per tile), runs a multi-level exclusive prefix sum over it, then does a stable scatter. Keys are ranked within a tile by a workgroup scan over sixteen one-hot counters packed two per 32-bit lane, so the output needs no atomics.
- **Bitonic sort.** A sorting network. Strides up to 256 run in shared memory inside 512-key tiles. Longer strides run in global memory, one dispatch per step. Sizes must be powers of two.

## Layout

| Path                 | Contents                                                                                                                                                                                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `index.html`, `src/` | The web demo: visualiser and benchmark                                                                                                                                                                                                                  |
| `kit/`               | [rc-webgpu-kit](https://github.com/rcarty/rc-webgpu-kit) as a git submodule. The WGSL sorts live in `kit/src/sort/`, shared with [rc-gsplat-mpm](https://github.com/rcarty/rc-gsplat-mpm), which uses the key-value radix sort to order splats by depth |
| `cuda/`              | The CUDA port and its benchmark                                                                                                                                                                                                                         |

## Web demo

```sh
git clone --recurse-submodules https://github.com/rcarty/rc-gpu-sort
cd rc-gpu-sort
python3 -m http.server        # then open http://localhost:8000
```

It needs a browser with WebGPU (recent Chrome, Edge, Safari or Firefox) and has no build step. The page has:

- a visualiser that sorts the shuffled pixels of a picture and animates them between passes
- a benchmark against `TypedArray.prototype.sort`, using timestamp queries where the browser allows them
- a correctness check of every GPU result against the CPU sort

The sorts have their own tests in rc-webgpu-kit (`kit/test/`).

## Formatting and lint

```sh
npm install
npm run lint      # eslint, then prettier --check
npm run format    # prettier --write
```

The rules come from rc-webgpu-kit (`kit/prettier.config.js`, `kit/eslint.config.js`), so every repository that uses it is formatted the same way. The CUDA code has its own `cuda/.clang-format`.

## CUDA

```sh
cd cuda
cmake -B build
cmake --build build
./build/gpu_sort                       # 64K, 1M, 4M and 16M keys, median of 5
./build/gpu_sort 16777216 --runs 10
./build/gpu_sort --csv > results.csv
```

On Windows, run these from a Visual Studio developer prompt, with `-G "NMake Makefiles"` on the first line. The binary is then `build\gpu_sort.exe`.

The CUDA kernels are straight ports of the WGSL: same tile sizes, digit width and scan structure. The benchmark also runs CUB's `DeviceRadixSort` and `DeviceScan` as references for what a tuned library achieves, with single-threaded `std::sort` and a plain loop as the CPU baselines. Input comes from the same xorshift32 generator and seed as the web page, so both sort identical keys. Every GPU result is checked against the CPU.

## Results

| Keys        | Radix | Bitonic | CUB  | CPU                  |
| ----------- | ----- | ------- | ---- | -------------------- |
| 1M, WebGPU  | 1.14  | 0.80    | –    | 30 (TypedArray.sort) |
| 1M, CUDA    | 1.13  | 0.71    | 0.16 | 66 (std::sort)       |
| 16M, WebGPU | 16.4  | 30.6    | –    | 564                  |
| 16M, CUDA   | 16.3  | 30.0    | 1.66 | 1266                 |

| Values      | Prefix sum | CUB  | CPU loop |
| ----------- | ---------- | ---- | -------- |
| 1M, WebGPU  | 0.05       | –    | 3.6      |
| 1M, CUDA    | 0.04       | 0.02 | 0.6      |
| 16M, WebGPU | 0.65       | –    | 57       |
| 16M, CUDA   | 0.59       | 0.26 | 10       |

GPU: NVIDIA RTX 2080 Ti. Times are medians in milliseconds, from GPU timestamp queries for the WebGPU rows (Chrome 158) and CUDA events for the CUDA rows (CUDA 13.5, MSVC 19.44, `--runs 30`). The WebGPU rows come from pressing "Run benchmark" several times in a row. The radix sort is slower for its first few runs on both APIs, and the 1M runs slow down by about 2.5 times once the GPU clocks down, so these are the figures once it has settled at full speed. The 16M `TypedArray.sort` figure is a single run.

## Next steps

- 8-bit digits with subgroup or warp ranking (`match_any`, ballots): four passes instead of eight
- A CUDA variant that uses warp intrinsics (`__shfl_up_sync`, `__ballot_sync`) for the in-tile scan and ranking, kept separate from the like-for-like port
- Subgroup operations in the WGSL version where the browser supports them
