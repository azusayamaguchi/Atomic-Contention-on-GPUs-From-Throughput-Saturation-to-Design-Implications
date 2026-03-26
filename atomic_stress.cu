#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

// --------------------
// Kernel
// --------------------
__global__ void atomic_stress(unsigned long long* data, int N, int K) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int target = idx % K;

    for (int i = 0; i < N; i++) {
        atomicAdd(&data[target], 1ULL);
    }
}

// --------------------
// Error check
// --------------------
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// --------------------
// Main
// --------------------
int main() {
    const int N = 100000;          // atomic iterations per thread
    const int threads_per_block = 256;

    // sweep blocks
    int block_list[] = {1, 2, 4, 8, 16, 32, 64, 128};
    int num_tests = sizeof(block_list) / sizeof(int);
    // hotspot 
    int hotspot_list[] = {1, 2, 4, 8, 16, 32, 64, 256};
    FILE* fp = fopen("atomic_stress_results_v3.csv", "w");
    if (!fp) {
        printf("Failed to open output CSV file.\n");
        return 1;
    }

    fprintf(fp,
            "hotspots,threads_per_block,blocks,total_threads,N,time_ms,throughput_ops_per_sec,sum\n");


    for (int h = 0; h < (int)(sizeof(hotspot_list)/sizeof(int)); h++) {

        int K = hotspot_list[h];
	unsigned long long* d_data;
	CHECK_CUDA(cudaMalloc(&d_data, K * sizeof(unsigned long long)));
        CHECK_CUDA(cudaMemset(d_data, 0, K * sizeof(unsigned long long)));

        printf("\n=== Hotspots = %d ===\n", K);


        for (int t = 0; t < num_tests; t++) {
            int blocks = block_list[t];
            int total_threads = blocks * threads_per_block;


            // reset data
	    CHECK_CUDA(cudaMemset(d_data, 0, K * sizeof(unsigned long long)));

            // timing
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);

            atomic_stress<<<blocks, threads_per_block>>>(d_data, N, K);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);

	    std::vector<unsigned long long> h_data(K);
            CHECK_CUDA(cudaMemcpy(h_data.data(), d_data, K * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

            unsigned long long sum = 0;
            for (int i = 0; i < K; i++) sum += h_data[i];

             double total_atomic_ops = (double)total_threads * N;
             double throughput = total_atomic_ops / (ms / 1000.0);

             printf("Blocks: %4d | Threads: %6d | Time: %8.3f ms | " "Throughput: %.2e ops/sec | Sum: %llu\n",
			          blocks, total_threads, ms, throughput, sum);

             fprintf(fp, "%d,%d,%d,%d,%d,%.6f,%.6e,%llu\n", K, threads_per_block, blocks, total_threads, N, ms, throughput, sum);

             cudaEventDestroy(start);
             cudaEventDestroy(stop);
	}
        cudaFree(d_data);
    }

    return 0;
}
