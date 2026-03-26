#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                   cudaGetErrorString(err));                                 \
            return 1;                                                        \
        }                                                                    \
    } while (0)

// Atomic hotspot microbenchmark.
// K controls the number of hotspot addresses.
// K=1   : worst contention
// K>1   : reduced contention
__global__ void atomic_stress(unsigned long long* data, int N, int K) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int target = idx % K;

    for (int i = 0; i < N; i++) {
        atomicAdd(&data[target], 1ULL);
    }
}

int main(int argc, char** argv) {
    // Default values
    int K = 1;
    int threads_per_block = 256;
    int total_threads = 32768;
    int N = 100000;

    // Usage:
    // ./atomic_stress_profile K tpb total_threads [N]
    if (argc >= 4) {
        K = std::atoi(argv[1]);
        threads_per_block = std::atoi(argv[2]);
        total_threads = std::atoi(argv[3]);
    }
    if (argc >= 5) {
        N = std::atoi(argv[4]);
    }

    if (K <= 0 || threads_per_block <= 0 || total_threads <= 0 || N <= 0) {
        printf("Usage: %s K threads_per_block total_threads [N]\n", argv[0]);
        printf("Example: %s 1 256 32768 100000\n", argv[0]);
        return 1;
    }

    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    printf("=============================================\n");
    printf("Atomic stress profile configuration\n");
    printf("Hotsptos(K)       = %d\n", K);
    printf("threads_per_block = %d\n", threads_per_block);
    printf("total_threads     = %d\n", total_threads);
    printf("blocks            = %d\n", blocks);
    printf("N                 = %d\n", N);
    printf("=============================================\n");


    // Sweep total thread counts
    //int total_threads_list[] = {1024, 2048, 4096, 8192, 16384, 32768};
    //int num_total_tests = sizeof(total_threads_list) / sizeof(int);

    // Sweep threads per block
    //int threads_per_block_list[] = {32, 64, 128, 256, 512};
    //int num_blocksize_tests = sizeof(threads_per_block_list) / sizeof(int);

    // Sweep hotspot counts
    //int hotspot_list[] = {1, 2, 4, 8, 16, 32, 64, 256};
    //int hotspot_list[] = {256};
    //int num_hotspot_tests = sizeof(hotspot_list) / sizeof(int);

    //FILE* fp = fopen("atomic_stress_results_v3.csv", "w");
    //if (!fp) {
    //    printf("Failed to open output CSV file.\n");
    //    return 1;
    //}

    //fprintf(fp, "hotspots,threads_per_block,blocks,total_threads,N,time_ms,throughput_ops_per_sec,sum\n");


    unsigned long long* d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, K * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_data, 0, K * sizeof(unsigned long long)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    atomic_stress<<<blocks, threads_per_block>>>(d_data, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    std::vector<unsigned long long> h_data(K);
    CHECK_CUDA(cudaMemcpy(h_data.data(),
                          d_data,
                          K * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    unsigned long long sum = 0;
    for (int i = 0; i < K; i++) {
        sum += h_data[i];
    }

    double total_atomic_ops = (double)total_threads * N;
    double throughput = total_atomic_ops / (ms / 1000.0);

    printf("Time (ms)         = %.6f\n", ms);
    printf("Throughput        = %.6e ops/sec\n", throughput);
    printf("Sum               = %llu\n", sum);
    printf("Expected sum      = %llu\n", (unsigned long long)total_threads * (unsigned long long)N);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_data));

    return 0;
}

