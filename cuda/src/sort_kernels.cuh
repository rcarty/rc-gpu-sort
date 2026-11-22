// straight ports of the WGSL sorts in rc-webgpu-kit (kit/src/sort): same tile size, digit width,
// scan structure and bitonic tiling, so the two APIs compare like for like.
// warp intrinsics (__shfl, __ballot) would be faster but belong in a separate variant
#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(x)                                                                    \
  do {                                                                                   \
    cudaError_t err_ = (x);                                                              \
    if (err_ != cudaSuccess) {                                                           \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

namespace gs {

constexpr int WG = 256;           // threads per block (WGSL workgroup_size)
constexpr int ITEMS = 4;          // keys per thread per tile
constexpr int TILE = WG * ITEMS;  // keys per block in the radix sort
constexpr int RADIX_BITS = 4;
constexpr int BINS = 1 << RADIX_BITS;
constexpr int PASSES = 32 / RADIX_BITS;
constexpr int BITONIC_TILE = 512;  // keys per block in the bitonic local kernel

// prefix sum

// exclusive scan of 256-element blocks (Hillis-Steele), block totals go to sums
__global__ void scanBlocks(uint32_t* __restrict__ data, uint32_t* __restrict__ sums, uint32_t n) {
  __shared__ uint32_t sh[WG];
  const uint32_t t = threadIdx.x;
  const uint32_t gi = blockIdx.x * WG + t;
  const uint32_t v = gi < n ? data[gi] : 0u;
  sh[t] = v;
  __syncthreads();
  for (uint32_t off = 1; off < WG; off <<= 1) {
    const uint32_t add = t >= off ? sh[t - off] : 0u;
    __syncthreads();
    sh[t] += add;
    __syncthreads();
  }
  if (gi < n) data[gi] = sh[t] - v;
  if (t == WG - 1) sums[blockIdx.x] = sh[WG - 1];
}

__global__ void addBlockSums(uint32_t* __restrict__ data, const uint32_t* __restrict__ sums,
                             uint32_t n) {
  const uint32_t gi = blockIdx.x * WG + threadIdx.x;
  if (gi < n) data[gi] += sums[blockIdx.x];
}

// in-place exclusive scan of n values. owns the upper levels
class Scanner {
 public:
  explicit Scanner(uint32_t n) {
    sizes_.push_back(n);
    while (sizes_.back() > WG) sizes_.push_back((sizes_.back() + WG - 1) / WG);
    for (size_t i = 1; i < sizes_.size(); ++i) {
      uint32_t* p = nullptr;
      CUDA_CHECK(cudaMalloc(&p, sizes_[i] * sizeof(uint32_t)));
      upper_.push_back(p);
    }
    CUDA_CHECK(cudaMalloc(&total_, sizeof(uint32_t)));
  }
  ~Scanner() {
    for (auto* p : upper_) cudaFree(p);
    cudaFree(total_);
  }
  Scanner(const Scanner&) = delete;
  Scanner& operator=(const Scanner&) = delete;

  void scan(uint32_t* data) const {
    const size_t levels = sizes_.size();
    auto level = [&](size_t i) { return i == 0 ? data : upper_[i - 1]; };
    auto blocks = [&](size_t i) { return (sizes_[i] + WG - 1) / WG; };
    for (size_t i = 0; i < levels; ++i)
      scanBlocks<<<blocks(i), WG>>>(level(i), i + 1 < levels ? level(i + 1) : total_, sizes_[i]);
    for (size_t i = levels - 1; i-- > 0;)
      addBlockSums<<<blocks(i), WG>>>(level(i), level(i + 1), sizes_[i]);
  }

 private:
  std::vector<uint32_t> sizes_;   // values per level, level 0 is the caller's data
  std::vector<uint32_t*> upper_;  // levels 1 and up
  uint32_t* total_ = nullptr;     // the top level's block sum lands here
};

// radix

// per-tile digit histogram, stored digit-major (hist[d * numBlocks + block]) so one scan
// over the whole array gives every block's offset per digit
__global__ void radixHist(const uint32_t* __restrict__ keys, uint32_t* __restrict__ hist,
                          uint32_t n, uint32_t shift, uint32_t numBlocks) {
  __shared__ uint32_t cnt[BINS];
  const uint32_t t = threadIdx.x;
  if (t < BINS) cnt[t] = 0;
  __syncthreads();
  const uint32_t base = blockIdx.x * TILE;
  for (int i = 0; i < ITEMS; ++i) {
    const uint32_t idx = base + i * WG + t;
    if (idx < n) atomicAdd(&cnt[(keys[idx] >> shift) & (BINS - 1)], 1u);
  }
  __syncthreads();
  if (t < BINS) hist[t * numBlocks + blockIdx.x] = cnt[t];
}

// stable scatter. keys are ranked in their 256-key chunk with a block-wide scan over 16 one-hot
// counters, packed two per 32-bit lane (8 lanes)
__global__ void radixScatter(const uint32_t* __restrict__ in, uint32_t* __restrict__ out,
                             const uint32_t* __restrict__ hist, uint32_t n, uint32_t shift,
                             uint32_t numBlocks) {
  __shared__ uint32_t scan[8][WG];  // lane-major to keep accesses bank-friendly
  __shared__ uint32_t base[BINS];
  const uint32_t t = threadIdx.x;
  if (t < BINS) base[t] = hist[t * numBlocks + blockIdx.x];
  __syncthreads();

  for (int i = 0; i < ITEMS; ++i) {
    const uint32_t idx = blockIdx.x * TILE + i * WG + t;
    const bool valid = idx < n;
    uint32_t key = 0, d = 0;
    if (valid) {
      key = in[idx];
      d = (key >> shift) & (BINS - 1);
    }
    for (int l = 0; l < 8; ++l) scan[l][t] = 0;
    if (valid) scan[d >> 1][t] = 1u << (16u * (d & 1u));
    __syncthreads();

    for (uint32_t off = 1; off < WG; off <<= 1) {
      uint32_t add[8];
      for (int l = 0; l < 8; ++l) add[l] = t >= off ? scan[l][t - off] : 0u;
      __syncthreads();
      for (int l = 0; l < 8; ++l) scan[l][t] += add[l];
      __syncthreads();
    }

    if (valid) {
      const uint32_t inc = scan[d >> 1][t];
      const uint32_t rank = ((inc >> (16u * (d & 1u))) & 0xFFFFu) - 1u;
      out[base[d] + rank] = key;
    }
    __syncthreads();
    if (t < BINS) base[t] += (scan[t >> 1][WG - 1] >> (16u * (t & 1u))) & 0xFFFFu;
    __syncthreads();
  }
}

// owns the histogram for one key count
class RadixSorter {
 public:
  explicit RadixSorter(uint32_t n)
      : n_(n), numBlocks_((n + TILE - 1) / TILE), scanner_(BINS * numBlocks_) {
    CUDA_CHECK(cudaMalloc(&hist_, BINS * numBlocks_ * sizeof(uint32_t)));
  }
  ~RadixSorter() { cudaFree(hist_); }
  RadixSorter(const RadixSorter&) = delete;
  RadixSorter& operator=(const RadixSorter&) = delete;

  // result ends in keys, tmp is scratch of the same size
  void sort(uint32_t* keys, uint32_t* tmp) const {
    for (int p = 0; p < PASSES; ++p) {
      const uint32_t* src = (p % 2 == 0) ? keys : tmp;
      uint32_t* dst = (p % 2 == 0) ? tmp : keys;
      const uint32_t shift = p * RADIX_BITS;
      radixHist<<<numBlocks_, WG>>>(src, hist_, n_, shift, numBlocks_);
      scanner_.scan(hist_);
      radixScatter<<<numBlocks_, WG>>>(src, dst, hist_, n_, shift, numBlocks_);
    }
  }

 private:
  uint32_t n_, numBlocks_;
  Scanner scanner_;
  uint32_t* hist_ = nullptr;
};

// bitonic

__global__ void bitonicGlobal(uint32_t* keys, uint32_t n, uint32_t k, uint32_t j) {
  const uint32_t i = blockIdx.x * WG + threadIdx.x;
  if (i >= n / 2) return;
  const uint32_t l = 2 * j * (i / j) + (i % j);
  const uint32_t r = l + j;
  const bool up = (l & k) == 0;
  const uint32_t x = keys[l], y = keys[r];
  if ((x > y) == up) {
    keys[l] = y;
    keys[r] = x;
  }
}

// k == 0: sort each 512-key tile completely (k = 2..512)
// k >= 1024: finish merge k for strides j = 256..1 inside the tile
__global__ void bitonicLocal(uint32_t* keys, uint32_t k) {
  __shared__ uint32_t sh[BITONIC_TILE];
  const uint32_t t = threadIdx.x;
  const uint32_t base = blockIdx.x * BITONIC_TILE;
  sh[t] = keys[base + t];
  sh[t + WG] = keys[base + t + WG];
  __syncthreads();
  const uint32_t kStart = k == 0 ? 2 : k, kEnd = k == 0 ? BITONIC_TILE : k;
  for (uint32_t kk = kStart; kk <= kEnd; kk <<= 1) {
    for (uint32_t j = min(kk / 2, 256u); j > 0; j >>= 1) {
      const uint32_t l = 2 * j * (t / j) + (t % j);
      const uint32_t r = l + j;
      const bool up = ((base + l) & kk) == 0;
      const uint32_t x = sh[l], y = sh[r];
      if ((x > y) == up) {
        sh[l] = y;
        sh[r] = x;
      }
      __syncthreads();
    }
  }
  keys[base + t] = sh[t];
  keys[base + t + WG] = sh[t + WG];
}

// n must be a power of two and at least 512
inline void bitonicSort(uint32_t* keys, uint32_t n) {
  const uint32_t globalBlocks = (n / 2 + WG - 1) / WG;
  const uint32_t localBlocks = n / BITONIC_TILE;
  bitonicLocal<<<localBlocks, WG>>>(keys, 0);
  for (uint32_t k = 2 * BITONIC_TILE; k <= n; k <<= 1) {
    for (uint32_t j = k / 2; j >= BITONIC_TILE; j >>= 1)
      bitonicGlobal<<<globalBlocks, WG>>>(keys, n, k, j);
    bitonicLocal<<<localBlocks, WG>>>(keys, k);
  }
}

}  // namespace gs
