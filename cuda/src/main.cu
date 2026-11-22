// radix, bitonic and prefix sum (ports of the WebGPU kernels) vs CUB and the CPU
//
//   gpu_sort [sizes...] [--runs N] [--csv]
//   gpu_sort 1048576 16777216 --runs 10
//
// same xorshift32 keys and seed as the web demo. one warm-up run, then the median of N,
// each on a fresh copy of the input. every result is checked against the CPU
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <functional>
#include <string>
#include <vector>

#include "sort_kernels.cuh"

static std::vector<uint32_t> randomKeys(uint32_t n, uint32_t seed) {
  std::vector<uint32_t> a(n);
  uint32_t x = seed ? seed : 1;
  for (uint32_t i = 0; i < n; ++i) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    a[i] = x;
  }
  return a;
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct Result {
  std::string name;
  double ms;
  bool ok;
};

// reset restores the input before each run
static double timeGPU(const std::function<void()>& reset, const std::function<void()>& run,
                      int runs) {
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  std::vector<double> times;
  for (int r = 0; r <= runs; ++r) {  // r == 0 is warm-up
    reset();
    CUDA_CHECK(cudaEventRecord(a));
    run();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    if (r > 0) times.push_back(ms);
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return median(times);
}

static bool matches(const uint32_t* dev, const std::vector<uint32_t>& ref) {
  std::vector<uint32_t> host(ref.size());
  CUDA_CHECK(cudaMemcpy(host.data(), dev, ref.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  return host == ref;
}

int main(int argc, char** argv) {
  std::vector<uint32_t> sizes;
  int runs = 5;
  bool csv = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--runs") && i + 1 < argc)
      runs = std::max(1, std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--csv"))
      csv = true;
    else
      sizes.push_back(static_cast<uint32_t>(std::strtoul(argv[i], nullptr, 10)));
  }
  if (sizes.empty()) sizes = {1u << 16, 1u << 20, 1u << 22, 1u << 24};

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  if (!csv)
    std::printf("GPU: %s\n\n", prop.name);
  else
    std::printf("gpu,keys,sort,median_ms,mkeys_per_s,matches_cpu\n");

  for (uint32_t n : sizes) {
    if (n < gs::BITONIC_TILE || (n & (n - 1))) {
      std::fprintf(stderr, "skipping %u: sizes must be powers of two >= %d\n", n, gs::BITONIC_TILE);
      continue;
    }
    const auto input = randomKeys(n, 987654321u);
    const size_t bytes = size_t(n) * sizeof(uint32_t);

    // CPU reference, single thread
    std::vector<double> cpuTimes;
    std::vector<uint32_t> ref;
    for (int r = 0; r < std::min(runs, 3); ++r) {
      ref = input;
      const auto t0 = std::chrono::high_resolution_clock::now();
      std::sort(ref.begin(), ref.end());
      const auto t1 = std::chrono::high_resolution_clock::now();
      cpuTimes.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    uint32_t *src, *keys, *tmp;
    CUDA_CHECK(cudaMalloc(&src, bytes));
    CUDA_CHECK(cudaMalloc(&keys, bytes));
    CUDA_CHECK(cudaMalloc(&tmp, bytes));
    CUDA_CHECK(cudaMemcpy(src, input.data(), bytes, cudaMemcpyHostToDevice));
    auto reset = [&] { CUDA_CHECK(cudaMemcpy(keys, src, bytes, cudaMemcpyDeviceToDevice)); };

    std::vector<Result> results;

    gs::RadixSorter radix(n);
    double ms = timeGPU(reset, [&] { radix.sort(keys, tmp); }, runs);
    results.push_back({"Radix (port of WebGPU)", ms, matches(keys, ref)});

    ms = timeGPU(reset, [&] { gs::bitonicSort(keys, n); }, runs);
    results.push_back({"Bitonic (port of WebGPU)", ms, matches(keys, ref)});

    size_t tempBytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, tempBytes, keys, tmp, int(n)));
    void* temp = nullptr;
    CUDA_CHECK(cudaMalloc(&temp, tempBytes));
    ms = timeGPU(
        reset,
        [&] { CUDA_CHECK(cub::DeviceRadixSort::SortKeys(temp, tempBytes, keys, tmp, int(n))); },
        runs);
    results.push_back({"CUB DeviceRadixSort", ms, matches(tmp, ref)});
    cudaFree(temp);

    results.push_back({"std::sort (CPU, 1 thread)", median(cpuTimes), true});

    // prefix sum over the same values, wrapping at 32 bits
    std::vector<uint32_t> sums(n);
    std::vector<double> loopTimes;
    for (int r = 0; r < 3; ++r) {
      const auto t0 = std::chrono::high_resolution_clock::now();
      uint32_t sum = 0;
      for (uint32_t i = 0; i < n; ++i) {
        sums[i] = sum;
        sum += input[i];
      }
      const auto t1 = std::chrono::high_resolution_clock::now();
      loopTimes.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    gs::Scanner scanner(n);
    ms = timeGPU(reset, [&] { scanner.scan(keys); }, runs);
    results.push_back({"Prefix sum (port of WebGPU)", ms, matches(keys, sums)});

    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, tempBytes, keys, tmp, int(n)));
    CUDA_CHECK(cudaMalloc(&temp, tempBytes));
    ms = timeGPU(
        reset,
        [&] { CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp, tempBytes, keys, tmp, int(n))); },
        runs);
    results.push_back({"CUB DeviceScan", ms, matches(tmp, sums)});
    cudaFree(temp);

    results.push_back({"Prefix sum loop (CPU)", median(loopTimes), true});

    if (!csv) std::printf("%u keys (median of %d)\n", n, runs);
    for (const auto& r : results) {
      const double mk = n / r.ms / 1000.0;
      if (csv)
        std::printf("\"%s\",%u,\"%s\",%.4f,%.1f,%d\n", prop.name, n, r.name.c_str(), r.ms, mk,
                    r.ok ? 1 : 0);
      else
        std::printf("  %-28s %10.3f ms %10.1f Mkeys/s  %s\n", r.name.c_str(), r.ms, mk,
                    r.ok ? "matches CPU" : "MISMATCH");
    }
    if (!csv) std::printf("\n");
    cudaFree(src);
    cudaFree(keys);
    cudaFree(tmp);
  }
  return 0;
}
