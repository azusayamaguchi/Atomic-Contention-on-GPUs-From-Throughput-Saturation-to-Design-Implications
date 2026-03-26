
import argparse
import glob
import math
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_TARGET_METRICS = {
    "L2 atomic active (avg)": "lts__d_atomic_input_cycles_active.avg",
    "L2 atomic active (sum)": "lts__d_atomic_input_cycles_active.sum",
    "LSU inst": "smsp__inst_executed_pipe_lsu.sum",
    "DRAM throughput": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "Stall (lg)": "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    "Stall (selected)": "smsp__warp_issue_stalled_selected_per_warp_active.pct",
    "Stall (membar)": "smsp__warp_issue_stalled_membar_per_warp_active.pct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Nsight Compute CSV files, compute mean/std by group, "
            "save summary tables, and generate comparison plots."
        )
    )
    parser.add_argument(
        "--k1-glob",
        default="ncu_resultsK1_*.csv",
        help="Glob pattern for the K=1 CSV files.",
    )
    parser.add_argument(
        "--k256-glob",
        default="ncu_resultsK256_*.csv",
        help="Glob pattern for the K=256 CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for summary tables and plots.",
    )
    parser.add_argument(
        "--summary-csv",
        default="ncu_summary_mean_std.csv",
        help="Filename for the combined mean±std summary table.",
    )
    parser.add_argument(
        "--save-run-tables",
        action="store_true",
        help="Save per-run metric tables for each group.",
    )
    parser.add_argument(
        "--plot-prefix",
        default="ncu",
        help="Prefix for generated plot filenames.",
    )
    parser.add_argument(
        "--include-selected",
        action="store_true",
        help="Include 'Stall (selected)' in plots.",
    )
    parser.add_argument(
        "--include-membar",
        action="store_true",
        help="Include 'Stall (membar)' in plots.",
    )
    return parser.parse_args()


def build_file_groups(args: argparse.Namespace) -> dict[str, list[str]]:
    return {
        "K=1": sorted(glob.glob(args.k1_glob)),
        "K=256": sorted(glob.glob(args.k256_glob)),
    }


def load_ncu_csv(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("==PROF=="):
                continue
            rows.append(line)

    df = pd.read_csv(StringIO("".join(rows)))

    df["Metric Value"] = (
        df["Metric Value"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("n/a", pd.NA)
    )
    df["Metric Value"] = pd.to_numeric(df["Metric Value"], errors="coerce")
    return df


def extract_metric_value(df: pd.DataFrame, metric_name: str) -> float:
    """Extract a metric value. If multiple rows exist, take the average."""
    sub = df[df["Metric Name"] == metric_name]
    if sub.empty:
        return math.nan
    return sub["Metric Value"].mean()


def aggregate_metrics(
    file_groups: dict[str, list[str]],
    target_metrics: dict[str, str],
    output_dir: Path,
    save_run_tables: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []

    for group_name, files in file_groups.items():
        if not files:
            print(f"[Warning] No files found for {group_name}")
            continue

        print(f"\nProcessing {group_name}:")
        for file_path in files:
            print(f"  {file_path}")

        per_run_records = []

        for file_path in files:
            df = load_ncu_csv(file_path)
            record = {"group": group_name, "file": file_path}

            for label, metric_name in target_metrics.items():
                record[label] = extract_metric_value(df, metric_name)

            per_run_records.append(record)

        runs_df = pd.DataFrame(per_run_records)

        for label in target_metrics.keys():
            values = pd.to_numeric(runs_df[label], errors="coerce")
            summary_rows.append(
                {
                    "group": group_name,
                    "metric": label,
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "num_runs": values.count(),
                }
            )

        if save_run_tables:
            run_table_name = f"{group_name.replace('=', '').replace(' ', '_')}_runs.csv"
            runs_df.to_csv(output_dir / run_table_name, index=False)

    summary_df = pd.DataFrame(summary_rows)

    combined_rows = []
    for metric in summary_df["metric"].unique():
        row = {"metric": metric}
        for group in summary_df["group"].unique():
            sub = summary_df[
                (summary_df["metric"] == metric) & (summary_df["group"] == group)
            ]
            if not sub.empty:
                mean = sub["mean"].iloc[0]
                std = sub["std"].iloc[0]
                row[group] = f"{mean:.6g} ± {std:.3g}"
        combined_rows.append(row)

    combined_df = pd.DataFrame(combined_rows)
    return summary_df, combined_df


def print_summary_tables(summary_df: pd.DataFrame) -> None:
    wide_mean = summary_df.pivot(index="metric", columns="group", values="mean")
    wide_std = summary_df.pivot(index="metric", columns="group", values="std")

    print("\n=== Mean ===")
    print(wide_mean)

    print("\n=== Std ===")
    print(wide_std)

    print("\n=== Mean ± Std ===")
    combined_rows = []
    for metric in summary_df["metric"].unique():
        row = {"metric": metric}
        for group in summary_df["group"].unique():
            sub = summary_df[
                (summary_df["metric"] == metric) & (summary_df["group"] == group)
            ]
            if not sub.empty:
                mean = sub["mean"].iloc[0]
                std = sub["std"].iloc[0]
                row[group] = f"{mean:.6g} ± {std:.3g}"
        combined_rows.append(row)

    combined_df = pd.DataFrame(combined_rows)
    print(combined_df.to_string(index=False))


def get_mean_std(summary_df: pd.DataFrame, metric: str, group: str) -> tuple[float, float]:
    sub = summary_df[(summary_df["metric"] == metric) & (summary_df["group"] == group)]
    if sub.empty:
        return math.nan, math.nan
    return float(sub["mean"].iloc[0]), float(sub["std"].iloc[0])


def plot_metric_comparison(
    summary_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    labels = ["K=1", "K=256"]
    means = []
    stds = []

    for group in labels:
        mean, std = get_mean_std(summary_df, metric, group)
        means.append(mean)
        stds.append(0.0 if pd.isna(std) else std)

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.bar(labels, means, yerr=stds, capsize=4)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_three_panel(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    labels = ["K=1", "K=256"]
    metric_specs = [
        ("L2 atomic active (avg)", "cycles", "L2 Atomic Activity"),
        ("LSU inst", "count", "LSU Instructions"),
        ("Stall (lg)", "%", "LSU Stall (lg)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (metric, ylabel, title) in zip(axes, metric_specs):
        means = []
        stds = []
        for group in labels:
            mean, std = get_mean_std(summary_df, metric, group)
            means.append(mean)
            stds.append(0.0 if pd.isna(std) else std)

        ax.bar(labels, means, yerr=stds, capsize=4)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_normalized(
    summary_df: pd.DataFrame,
    metrics: list[str],
    output_path: Path,
    title: str = "Relative to K=256 (K=1 / K=256)",
) -> None:
    values = []
    labels = []

    for metric in metrics:
        k1_mean, _ = get_mean_std(summary_df, metric, "K=1")
        k256_mean, _ = get_mean_std(summary_df, metric, "K=256")
        if pd.isna(k1_mean) or pd.isna(k256_mean) or k256_mean == 0:
            continue

        labels.append(metric)
        values.append(k1_mean / k256_mean)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylabel("Relative value")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_groups = build_file_groups(args)
    target_metrics = dict(DEFAULT_TARGET_METRICS)

    summary_df, combined_df = aggregate_metrics(
        file_groups=file_groups,
        target_metrics=target_metrics,
        output_dir=output_dir,
        save_run_tables=args.save_run_tables,
    )

    if summary_df.empty:
        raise SystemExit("No valid input files were found.")

    print_summary_tables(summary_df)

    summary_csv_path = output_dir / args.summary_csv
    combined_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved summary table to: {summary_csv_path}")

    # Main plots
    plot_metric_comparison(
        summary_df,
        metric="L2 atomic active (avg)",
        ylabel="cycles",
        title="L2 Atomic Activity",
        output_path=output_dir / f"{args.plot_prefix}_atomic.png",
    )
    plot_metric_comparison(
        summary_df,
        metric="LSU inst",
        ylabel="count",
        title="LSU Instructions",
        output_path=output_dir / f"{args.plot_prefix}_lsu.png",
    )
    plot_metric_comparison(
        summary_df,
        metric="Stall (lg)",
        ylabel="%",
        title="LSU Stall (lg)",
        output_path=output_dir / f"{args.plot_prefix}_stall_lg.png",
    )

    plot_three_panel(
        summary_df=summary_df,
        output_path=output_dir / f"{args.plot_prefix}_three_panel.png",
    )

    normalized_metrics = ["L2 atomic active (avg)", "LSU inst", "Stall (lg)"]
    if args.include_selected:
        normalized_metrics.append("Stall (selected)")
    if args.include_membar:
        normalized_metrics.append("Stall (membar)")

    plot_normalized(
        summary_df=summary_df,
        metrics=normalized_metrics,
        output_path=output_dir / f"{args.plot_prefix}_normalized.png",
    )

    print("\nSaved plots:")
    print(f"  {output_dir / f'{args.plot_prefix}_atomic.png'}")
    print(f"  {output_dir / f'{args.plot_prefix}_lsu.png'}")
    print(f"  {output_dir / f'{args.plot_prefix}_stall_lg.png'}")
    print(f"  {output_dir / f'{args.plot_prefix}_three_panel.png'}")
    print(f"  {output_dir / f'{args.plot_prefix}_normalized.png'}")


if __name__ == "__main__":
    main()
