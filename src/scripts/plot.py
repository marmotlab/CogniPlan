#!/usr/bin/env python3
"""Plot comparison of metrics files.

Each metrics_*.txt file has four columns:
1) explored volume (m^3)
2) traveling distance (m)
3) algorithm runtime (s)
4) time since start (s)
"""
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(file_path: Path) -> np.ndarray:
	data = np.loadtxt(file_path)
	if data.ndim == 1:
		data = data.reshape(1, -1)
	if data.shape[1] < 4:
		raise ValueError(f"{file_path} has {data.shape[1]} columns; expected 4.")
	return data[:, :4]


def compute_completion_point(data: np.ndarray, distance_eps: float = 1e-6) -> Tuple[float, float]:
	distances = data[:, 1]
	times = data[:, 3]

	if len(distances) < 2:
		return float(times[-1]), float(distances[-1])

	diff = np.abs(np.diff(distances))
	change_indices = np.where(diff > distance_eps)[0]
	if change_indices.size == 0:
		completion_idx = len(distances) - 1
	else:
		completion_idx = int(change_indices[-1] + 1)

	return float(times[completion_idx]), float(distances[completion_idx])


def main() -> None:
	scripts_dir = Path(__file__).resolve().parent
	metric_files = sorted(scripts_dir.glob("metrics_*.txt"))

	if not metric_files:
		raise SystemExit("No metrics_*.txt files found in scripts directory.")

	fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
	axes = axes.flatten()

	titles = [
		"Explored Volume (m^3)",
		"Traveling Distance (m)",
		"Algorithm Runtime (s)",
		"Completion (Time vs Distance)",
	]
	ylabels = [
		"Explored Volume (m^3)",
		"Distance (m)",
		"Runtime (s)",
		"Total Distance (m)",
	]

	color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
	for idx, file_path in enumerate(metric_files):
		data = load_metrics(file_path)
		label = file_path.stem
		x = data[:, 3]
		color = color_cycle[idx % len(color_cycle)] if color_cycle else None
		for i in range(3):
			axes[i].plot(x, data[:, i], label=label, linewidth=1.5, color=color)

		completion_time, completion_distance = compute_completion_point(data)
		axes[3].scatter(
			completion_time,
			completion_distance,
			label=label,
			s=60,
			color=color,
			edgecolor="black",
			linewidth=0.5,
		)

	for i, ax in enumerate(axes):
		ax.set_title(titles[i])
		if i < 3:
			ax.set_xlabel("Elapsed Time (s)")
		else:
			ax.set_xlabel("Completion Time (s)")
		ax.set_ylabel(ylabels[i])
		ax.grid(True, linestyle="--", alpha=0.4)

	handles, labels = axes[0].get_legend_handles_labels()
	if handles:
		fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

	fig.tight_layout(rect=[0, 0, 1, 0.93])
	output_path = scripts_dir / "metrics_comparison.png"
	fig.savefig(output_path, dpi=200)
	plt.show()


if __name__ == "__main__":
	main()
