// Read bandwidth vs working-set size: the L2 plateau, the L2 -> DRAM cliff, and DRAM.
//
//   nvcc -O3 -arch=sm_87 -o l2_bandwidth l2_bandwidth.cu      (RTX 4090: -arch=sm_89)
//   ./l2_bandwidth [max_mib=512] > l2_bandwidth.csv
//
// Every thread streams float4 loads with __ldcg (cache in L2 only, bypass L1), so a buffer smaller
// than the L2 is served from L2 after the first pass and a larger one from DRAM. Each size is read
// repeatedly until ~4 GiB have moved; the best of 5 timed runs is reported. Output: CSV
// size_mib,gbps.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void read_kernel(const float4 * __restrict__ p, size_t n, int iters, float * out) {
    float4 acc = make_float4(0, 0, 0, 0);
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    for (int it = 0; it < iters; ++it) {
        for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
            const float4 v = __ldcg(p + i);
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
    }
    if (acc.x + acc.y + acc.z + acc.w == 1234.5f) {   // never true; keeps the loads alive
        out[0] = acc.x;
    }
}

int main(int argc, char ** argv) {
    const double max_mib = argc > 1 ? atof(argv[1]) : 512.0;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "%s: %d SMs, L2 %d KiB\n", prop.name, prop.multiProcessorCount, prop.l2CacheSize / 1024);

    const int threads = 256;
    const int blocks  = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / threads);
    const size_t max_bytes = (size_t) (max_mib * 1024 * 1024);
    float4 * buf; float * out;
    CK(cudaMalloc(&buf, max_bytes));
    CK(cudaMalloc(&out, sizeof(float)));
    CK(cudaMemset(buf, 0, max_bytes));

    const double sizes_mib[] = {0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 72,
                                96, 128, 256, 512};
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    printf("size_mib,gbps\n");
    for (double mib : sizes_mib) {
        const size_t bytes = (size_t) (mib * 1024 * 1024);
        if (bytes > max_bytes) {
            break;
        }
        const size_t n = bytes / sizeof(float4);
        const int iters = std::max(1, (int) ((4.0 * 1024 * 1024 * 1024) / bytes));
        read_kernel<<<blocks, threads>>>(buf, n, 1, out);   // warm: bring the buffer into L2
        CK(cudaDeviceSynchronize());
        float best = 1e30f;
        for (int rep = 0; rep < 5; ++rep) {
            CK(cudaEventRecord(t0));
            read_kernel<<<blocks, threads>>>(buf, n, iters, out);
            CK(cudaEventRecord(t1));
            CK(cudaEventSynchronize(t1));
            float ms;
            CK(cudaEventElapsedTime(&ms, t0, t1));
            best = std::min(best, ms);
        }
        printf("%g,%.1f\n", mib, (double) bytes * iters / (best * 1e-3) / 1e9);
        fflush(stdout);
    }
    return 0;
}
