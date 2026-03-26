import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "atomic_stress_results_v3.csv"

df = pd.read_csv(CSV_FILE)

# 1) Hotspotごとの throughput curve
for tpb in sorted(df["threads_per_block"].unique()):
    sub = df[df["threads_per_block"] == tpb]

    plt.figure(figsize=(8, 5))
    for k in sorted(sub["hotspots"].unique()):
        s = sub[sub["hotspots"] == k].sort_values("total_threads")
        plt.plot(
            s["total_threads"],
            s["throughput_ops_per_sec"],
            marker="o",
            label=f"K={k}"
        )

    plt.xlabel("Total threads")
    plt.ylabel("Throughput (ops/sec)")
    plt.title(f"Atomic throughput vs total threads (threads_per_block={tpb})")
    plt.xscale("log", base=2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"throughput_by_hotspot_tpb_{tpb}.png", dpi=150)
    plt.close()

# 2) threads_per_blockごとの比較（hotspot固定）
for k in sorted(df["hotspots"].unique()):
    sub = df[df["hotspots"] == k]

    plt.figure(figsize=(8, 5))
    for tpb in sorted(sub["threads_per_block"].unique()):
        s = sub[sub["threads_per_block"] == tpb].sort_values("total_threads")
        plt.plot(
            s["total_threads"],
            s["throughput_ops_per_sec"],
            marker="o",
            label=f"tpb={tpb}"
        )

    plt.xlabel("Total threads")
    plt.ylabel("Throughput (ops/sec)")
    plt.title(f"Atomic throughput vs total threads (hotspots={k})")
    plt.xscale("log", base=2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"throughput_by_tpb_hotspot_{k}.png", dpi=150)
    plt.close()

# 3) 時間の比較も保存
for k in sorted(df["hotspots"].unique()):
    sub = df[df["hotspots"] == k]

    plt.figure(figsize=(8, 5))
    for tpb in sorted(sub["threads_per_block"].unique()):
        s = sub[sub["threads_per_block"] == tpb].sort_values("total_threads")
        plt.plot(
            s["total_threads"],
            s["time_ms"],
            marker="o",
            label=f"tpb={tpb}"
        )

    plt.xlabel("Total threads")
    plt.ylabel("Kernel time (ms)")
    plt.title(f"Kernel time vs total threads (hotspots={k})")
    plt.xscale("log", base=2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"time_by_tpb_hotspot_{k}.png", dpi=150)
    plt.close()

print("Plots saved as PNG files.")

