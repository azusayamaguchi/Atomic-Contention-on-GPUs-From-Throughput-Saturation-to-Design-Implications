GPU Atomic Contention: A Microbenchmark Study

This project investigates the scalability limits of atomic operations on GPUs (NVIDIA RTX A2000) by quantifying the impact of thread-level parallelism and hotspot density (contention).

The goal is to demonstrate that, unlike CPUs where performance degradation is driven by coherence-induced latency, GPU atomic-heavy workloads are limited by throughput saturation due to atomic serialization.

    Methodology

We implemented a CUDA microbenchmark where each thread repeatedly performs atomic additions to a limited set of memory addresses:

atomicAdd(&data[idx % K], 1ULL);

Parameters

N=100000: number of atomic operations per thread threads_per_block: 32, 64, 128, 256, 512 total_threads: 1024 to 32768 K={1,2,4,8,16,32,64,256}: number of hotspots

1.1 Contention Control

Each thread computes:

target=idx mod K

K=1: all threads access the same address → maximum contention K>1: accesses are distributed → reduced contention K=256: near contention-free

    Results 2.1 Effect of Block Size (tpb)

Since GPUs schedule execution at warp granularity (32 threads), threads_per_block affects warp-level parallelism and scheduling behavior.

tpb = 128 (4 warps): insufficient parallelism → latency not hidden tpb = 256 (8 warps): best balance → stable performance tpb = 512 (16 warps): burst contention and reduced scheduling flexibility

2.2 Hotspot Behavior

Throughput for K=1 and K=2 is nearly identical Performance improves only when K≥4

This indicates: Logical distribution does not guarantee physical parallelism. Effective parallelism is limited by hardware resources (L2 slices / memory partitions).

2.3 Throughput Saturation

Increasing thread count leads to early saturation (plateau):

Small K (high contention) → low ceiling Large K (low contention) → high ceiling

This behavior is well described by: Throughput=min⁡(arrival rate,service rate) Here, arrival rate ≈ threads × N service rate ≈ atomic pipeline capacity

2.4 Nsight Systems Profiling

Profiling confirms that the bottleneck lies entirely within the GPU kernel:

Kernel execution dominates runtime Memory transfers and launch overhead are negligible K=256 is ~11.6× faster than K=1

2.5 Nsight Compute Analysis

We collected the following metrics:

L2 atomic activity LSU instruction count Warp stall (lg / membar / selected) DRAM throughput Key Observations L2 atomic activity: 4× higher at K=1 LSU instructions: identical DRAM throughput: ~0 Stall (lg): ~97% (slightly higher at K=1)

Interpretation Instruction count is identical → not compute-bound DRAM usage is negligible → not memory-bandwidth-bound Atomic activity increases significantly → atomic pipeline pressure Warp execution is almost always stalled → strong backpressure

Therefore, the performance difference is caused by contention-induced atomic serialization.

    Conclusion

GPU atomic performance is not determined by individual memory latency, but by the throughput of the atomic pipeline.

Small K: serialization → low service rate → low throughput Large K: distributed accesses → higher service rate → higher throughput

Thus, GPU atomic-heavy workloads are fundamentally throughput-bound, and the primary bottleneck is atomic serialization.

    CPU vs GPU Comparison

This behavior differs fundamentally from CPUs:

CPU (e.g., many core ) Coherence traffic and ownership migration → latency amplification → performance collapse

GPU Atomic serialization → throughput saturation

CPU: latency-bound (coherence-driven) GPU: throughput-bound (serialization-driven)

This highlights a fundamental difference in scalability failure modes between CPU and GPU architectures.
