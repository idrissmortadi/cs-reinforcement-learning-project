import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Apply Seaborn style
sns.set_theme(style="whitegrid")

NUM_COLS = 3  # Number of columns in the grid plot


def smooth_curve(values, weight=0.9):
    """
    Smooths a curve using exponential moving average.

    Args:
        values (list): List of values to smooth.
        weight (float): Smoothing weight (0 < weight < 1).

    Returns:
        list: Smoothed values.
    """
    smoothed = []
    last = values[0]
    for value in values:
        smoothed_value = last * weight + (1 - weight) * value
        smoothed.append(smoothed_value)
        last = smoothed_value
    return smoothed


def export_figures_from_event_file(event_file, run_name):
    """
    Extracts all figures from a TensorFlow event file and saves them in results/figures/RUN_NAME.

    Args:
        event_file (str): Path to the TensorFlow event file.
        run_name (str): Name of the run to use for the output directory.
    """
    # Define output directory
    output_dir = os.path.join("results", "figures", run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load the event file
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    # Collect scalar tags and their data
    scalar_data = []
    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        smoothed_values = smooth_curve(values)
        scalar_data.append((tag, steps, values, smoothed_values))

        # Plot individual scalar values using Seaborn
        plt.figure(figsize=(6, 6))
        sns.lineplot(
            x=steps, y=values, label=f"{tag} (original)", alpha=0.3, color="blue"
        )
        sns.lineplot(
            x=steps, y=smoothed_values, label=f"{tag} (smoothed)", color="blue"
        )
        plt.xlabel("Steps", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.title(tag, fontsize=16)
        plt.legend(fontsize=12)
        plt.tight_layout()

        # Save the figure
        figure_path = os.path.join(output_dir, f"{tag.replace('/', '_')}.png")
        plt.savefig(figure_path)
        plt.close()

    # Create a grid of plots for all metrics
    num_metrics = len(scalar_data)
    num_rows = (num_metrics + NUM_COLS - 1) // NUM_COLS  # Calculate rows needed
    fig, axes = plt.subplots(num_rows, NUM_COLS, figsize=(NUM_COLS * 6, num_rows * 4))
    axes = axes.flatten()

    for i, (tag, steps, values, smoothed_values) in enumerate(scalar_data):
        sns.lineplot(
            ax=axes[i], x=steps, y=values, label="Original", alpha=0.3, color="blue"
        )
        sns.lineplot(
            ax=axes[i], x=steps, y=smoothed_values, label="Smoothed", color="blue"
        )
        axes[i].set_title(tag, fontsize=14)
        axes[i].set_xlabel("Steps", fontsize=12)
        axes[i].set_ylabel("Value", fontsize=12)
        axes[i].legend(fontsize=10)
        axes[i].grid(True)

    # Hide unused subplots
    for j in range(len(scalar_data), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    grid_figure_path = os.path.join(output_dir, "all_metrics_grid.png")
    plt.savefig(grid_figure_path)
    plt.close()

    print(f"Figures exported to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export figures from a TensorFlow event file."
    )
    parser.add_argument(
        "event_file", type=str, help="Path to the TensorFlow event file."
    )
    parser.add_argument(
        "run_name", type=str, help="Name of the run to use for the output directory."
    )
    args = parser.parse_args()

    export_figures_from_event_file(args.event_file, args.run_name)
