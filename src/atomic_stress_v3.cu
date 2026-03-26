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

int main() {
    const int N = 100000;  // atomic iterations per thread

    // Sweep total thread counts
    int total_threads_list[] = {1024, 2048, 4096, 8192, 16384, 32768};
    int num_total_tests = sizeof(total_threads_list) / sizeof(int);

    // Sweep threads per block
    int threads_per_block_list[] = {32, 64, 128, 256, 512};
    int num_blocksize_tests = sizeof(threads_per_block_list) / sizeof(int);

    // Sweep hotspot counts
    int hotspot_list[] = {1, 2, 4, 8, 16, 32, 64, 256};
    int num_hotspot_tests = sizeof(hotspot_list) / sizeof(int);

    FILE* fp = fopen("atomic_stress_results_v3.csv", "w");
    if (!fp) {
        printf("Failed to open output CSV file.\n");
        return 1;
    }

    fprintf(fp, "hotspots,threads_per_block,blocks,total_threads,N,time_ms,throughput_ops_per_sec,sum\n");


    for (int h = 0; h < num_hotspot_tests; h++) {
        int K = hotspot_list[h];

        unsigned long long* d_data = nullptr;
        CHECK_CUDA(cudaMalloc(&d_data, K * sizeof(unsigned long long)));

        printf("\n============================================================\n");
        printf("Hotspots = %d\n", K);
        printf("============================================================\n");

        for (int b = 0; b < num_blocksize_tests; b++) {
            int threads_per_block = threads_per_block_list[b];

            printf("\n--- Threads per block = %d ---\n", threads_per_block);

            for (int t = 0; t < num_total_tests; t++) {
                int total_threads = total_threads_list[t];
                int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

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

                printf("Blocks: %4d | Threads: %6d | Time: %8.3f ms | " "Throughput: %.2e ops/sec | Sum: %llu\n",
                       blocks, total_threads, ms, throughput, sum);

		fprintf(fp, "%d,%d,%d,%d,%d,%.6f,%.6e,%llu\n",
                        K, threads_per_block, blocks, total_threads, N, ms, throughput, sum);

                CHECK_CUDA(cudaEventDestroy(start));
                CHECK_CUDA(cudaEventDestroy(stop));
            }
        }

        CHECK_CUDA(cudaFree(d_data));
    }

    return 0;
}

