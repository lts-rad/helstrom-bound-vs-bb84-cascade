"""Run real partial-interception attacks and plot hybrid recovery in 3D."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np

from hybrid_constraint_solver import HybridConstraintSolver, score_result
from quadrature_attack_model import QuadratureAttackModel


DEFAULT_RAW_SIZES = (2048, 8192, 32768, 131072, 400000)
DEFAULT_SAMPLING_RATES = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_SEEDS = (42,)


class PartialSamplingModel(QuadratureAttackModel):
    """Intercept-resend model in which Eve samples only some transmissions."""

    def __init__(self, raw_key_size: int, seed: int, sampling_rate: float):
        if not 0.0 <= sampling_rate <= 1.0:
            raise ValueError("sampling_rate must be between 0 and 1")

        super().__init__(raw_key_size, seed)
        self.sampling_rate = sampling_rate

        if sampling_rate == 0.0:
            sampled_raw = np.zeros(raw_key_size, dtype=bool)
        elif sampling_rate == 1.0:
            sampled_raw = np.ones(raw_key_size, dtype=bool)
        else:
            # Use a separate generator so the sampling choice does not alter the
            # underlying Alice/Bob/Eve realization for a given seed.
            rng = np.random.default_rng(seed + 1_000_003)
            sampled_raw = rng.random(raw_key_size) < sampling_rate

        self.eve_sampled_raw = sampled_raw.tolist()
        self.eve_sampled_sifted = []

        for sifted_index, raw_index in enumerate(self.sifted_indices):
            sampled = bool(sampled_raw[raw_index])
            self.eve_sampled_sifted.append(sampled)
            if sampled:
                continue

            # An unsampled photon reaches Bob unchanged. Eve knows that she did
            # not sample this position, but has no value to give the decoder.
            alice_bit = self.alice_bits[sifted_index]
            self.bob_bits[sifted_index] = alice_bit
            self.bob_bits_raw[raw_index] = alice_bit
            self.eve_measurements[sifted_index] = 0
            self.eve_measurements_raw[raw_index] = 0
            self.eve_had_correct_basis[sifted_index] = False
            self.eve_had_correct_basis_raw[raw_index] = False

        self.bob_key._bits = self.bob_bits.copy()
        self.realized_sifted_sampling_rate = (
            sum(self.eve_sampled_sifted) / self.sifted_key_size
            if self.sifted_key_size
            else 0.0
        )


def run_attack(raw_size: int, sampling_rate: float, seed: int) -> dict[str, float | int | bool]:
    """Run one CASCADE transcript and solve it using only the hybrid decoder."""
    model = PartialSamplingModel(raw_size, seed, sampling_rate)
    with contextlib.redirect_stdout(io.StringIO()):
        constraints = model.run_cascade()

    result = HybridConstraintSolver.from_model(model, constraints).solve()
    score = score_result(model, result)
    return {
        "raw_size": raw_size,
        "sifted_size": model.sifted_key_size,
        "seed": seed,
        "sampling_rate": sampling_rate,
        "realized_sampling_rate": model.realized_sifted_sampling_rate,
        "qber": model.calculate_qber(),
        "accuracy": score["accuracy"],
        "baseline_accuracy": score["baseline_accuracy"],
        "remaining_errors": score["remaining_errors"],
        "constraints": len(constraints),
        "rank": result.rank,
        "violations": result.affine_violations,
        "optimal": result.optimal,
        "solve_seconds": result.total_seconds,
        "z3_checks": result.z3_checks,
    }


def run_sweep(raw_sizes, sampling_rates, seeds):
    rows = []
    print(
        "raw sift sample realized qber accuracy errors rows rank solve_s "
        "optimal seed"
    )
    for raw_size in raw_sizes:
        for sampling_rate in sampling_rates:
            for seed in seeds:
                row = run_attack(raw_size, sampling_rate, seed)
                rows.append(row)
                print(
                    f"{raw_size:6d} {row['sifted_size']:6d} "
                    f"{sampling_rate:6.1%} {row['realized_sampling_rate']:7.2%} "
                    f"{row['qber']:6.2%} {row['accuracy']:8.3%} "
                    f"{row['remaining_errors']:6d} {row['constraints']:6d} "
                    f"{row['rank']:6d} {row['solve_seconds']:7.3f} "
                    f"{str(row['optimal']):>7} {seed:4d}"
                )
    return rows


def aggregate(rows, raw_sizes, sampling_rates):
    """Average repeated seeds while retaining the rectangular sweep grid."""
    metrics = (
        "sifted_size",
        "realized_sampling_rate",
        "qber",
        "accuracy",
        "baseline_accuracy",
        "remaining_errors",
        "constraints",
        "rank",
        "solve_seconds",
    )
    aggregated = []
    for raw_size in raw_sizes:
        for sampling_rate in sampling_rates:
            selected = [
                row
                for row in rows
                if row["raw_size"] == raw_size
                and row["sampling_rate"] == sampling_rate
            ]
            if not selected:
                raise ValueError("incomplete partial-sampling sweep")
            item = {"raw_size": raw_size, "sampling_rate": sampling_rate}
            for metric in metrics:
                item[metric] = float(np.mean([row[metric] for row in selected]))
            item["optimal"] = all(bool(row["optimal"]) for row in selected)
            item["violations"] = max(int(row["violations"]) for row in selected)
            aggregated.append(item)
    return aggregated


def plot_results(rows, raw_sizes, sampling_rates, output_path: Path):
    averaged = aggregate(rows, raw_sizes, sampling_rates)
    by_point = {
        (row["raw_size"], row["sampling_rate"]): row for row in averaged
    }

    sifted = np.array(
        [
            [by_point[(size, rate)]["sifted_size"] for rate in sampling_rates]
            for size in raw_sizes
        ]
    )
    qber = np.array(
        [
            [by_point[(size, rate)]["qber"] * 100 for rate in sampling_rates]
            for size in raw_sizes
        ]
    )
    accuracy = np.array(
        [
            [by_point[(size, rate)]["accuracy"] * 100 for rate in sampling_rates]
            for size in raw_sizes
        ]
    )
    sample_grid = np.tile(np.asarray(sampling_rates), (len(raw_sizes), 1))

    x = np.log2(sifted)
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = cm.viridis

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x,
        qber,
        accuracy,
        facecolors=cmap(norm(sample_grid)),
        edgecolor="0.35",
        linewidth=0.35,
        antialiased=True,
        alpha=0.72,
        shade=False,
    )
    ax.scatter(
        x.ravel(),
        qber.ravel(),
        accuracy.ravel(),
        c=sample_grid.ravel(),
        cmap=cmap,
        norm=norm,
        s=48,
        edgecolors="black",
        linewidths=0.45,
        depthshade=False,
    )

    for size_index in range(len(raw_sizes)):
        ax.plot(
            x[size_index],
            qber[size_index],
            accuracy[size_index],
            color="0.3",
            linewidth=0.8,
            alpha=0.65,
        )

    mean_sifted = np.mean(sifted, axis=1)
    ax.set_xticks(np.log2(mean_sifted))
    ax.set_xticklabels([f"{value / 1000:.1f}k" for value in mean_sifted])
    ax.set_xlabel("Mean sifted key size (bits)", labelpad=12)
    ax.set_ylabel("Measured Alice–Bob QBER (%)", labelpad=12)
    ax.set_zlabel("Hybrid recovery accuracy (%)", labelpad=12)
    ax.set_title("Hybrid CASCADE attack under partial interception", pad=18)
    ax.set_zlim(45, 101)
    ax.view_init(elev=25, azim=-55)

    colorbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        shrink=0.68,
        pad=0.1,
    )
    colorbar.set_label("Eve sampling rate")
    colorbar.set_ticks(sampling_rates)
    colorbar.set_ticklabels([f"{rate:.0%}" for rate in sampling_rates])

    # Matplotlib's tight_layout does not handle 3D axes and colorbars reliably.
    fig.subplots_adjust(left=0.02, right=0.82, bottom=0.06, top=0.91)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def write_csv(rows, path: Path):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-sizes", nargs="+", type=int, default=DEFAULT_RAW_SIZES)
    parser.add_argument(
        "--sampling-rates", nargs="+", type=float, default=DEFAULT_SAMPLING_RATES
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output", type=Path, default=Path("partial_sampling_hybrid_3d.png"))
    parser.add_argument(
        "--csv", type=Path, default=Path("partial_sampling_hybrid_results.csv")
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    raw_sizes = tuple(args.raw_sizes)
    sampling_rates = tuple(sorted(args.sampling_rates))
    seeds = tuple(args.seeds)
    if len(raw_sizes) < 2 or len(sampling_rates) < 2:
        parser.error("the 3D plot needs at least two sizes and two sampling rates")

    rows = run_sweep(raw_sizes, sampling_rates, seeds)
    write_csv(rows, args.csv)
    plot_results(rows, raw_sizes, sampling_rates, args.output)
    print(f"Saved plot to {args.output}")
    print(f"Saved data to {args.csv}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
