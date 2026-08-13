"""Plot hybrid key recovery for variable-μ partial interception."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from hybrid_constraint_solver import HybridConstraintSolver, score_result
from qpsk_srm_attack import (
    PartialInterceptMeanPhotonAttackModel,
    srm_probabilities,
)


DEFAULT_QBER_LIMIT = 0.11
DEFAULT_RAW_SIZE = 32_768
DEFAULT_SEEDS = (42,)


def expected_attack_rates(
    mean_photon_number: float, intercept_fraction: float
) -> dict[str, float]:
    """Expected attack rates with unintercepted signals passed through."""
    if not 0.0 <= intercept_fraction <= 1.0:
        raise ValueError("intercept_fraction must be between 0 and 1")
    probabilities = srm_probabilities(mean_photon_number)
    return {
        "hber_intercepted": probabilities.hber,
        "qber_intercepted": probabilities.qber,
        "hber_effective": intercept_fraction * probabilities.hber,
        "qber_effective": intercept_fraction * probabilities.qber,
    }


def maximum_intercept_fraction(
    mean_photon_number: float, qber_limit: float = DEFAULT_QBER_LIMIT
) -> float:
    """Largest intercepted fraction whose expected attack QBER meets a limit."""
    if not 0.0 <= qber_limit <= 0.5:
        raise ValueError("qber_limit must be between 0 and 0.5")
    full_qber = srm_probabilities(mean_photon_number).qber
    if full_qber == 0.0:
        return 1.0
    return min(1.0, qber_limit / full_qber)


def run_recovery_attack(
    raw_size: int,
    mean_photon_number: float,
    intercept_fraction: float,
    seed: int,
) -> dict[str, float | int | bool | str]:
    """Run one real CASCADE transcript and score the hybrid reconstruction."""
    model = PartialInterceptMeanPhotonAttackModel(
        raw_size,
        seed,
        mean_photon_number,
        intercept_fraction,
    )

    # With no interception Bob's sifted key is already clean, so error
    # correction has no public transcript to exploit.
    if intercept_fraction == 0.0:
        constraints = []
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            constraints = model.run_cascade()

    result = HybridConstraintSolver.from_model(model, constraints).solve()
    score = score_result(model, result)
    expected = expected_attack_rates(mean_photon_number, intercept_fraction)
    return {
        "raw_size": raw_size,
        "sifted_size": model.sifted_key_size,
        "seed": seed,
        "mean_photon_number": mean_photon_number,
        "intercept_fraction": intercept_fraction,
        "realized_intercept_fraction": model.realized_sifted_intercept_fraction,
        "expected_qber": expected["qber_effective"],
        "measured_qber": model.calculate_qber(),
        "recovery_accuracy": score["accuracy"],
        "baseline_accuracy": score["baseline_accuracy"],
        "remaining_errors": score["remaining_errors"],
        "constraints": len(constraints),
        "rank": result.rank,
        "solve_seconds": result.total_seconds,
        "optimal": result.optimal,
        "status": result.status,
    }


def run_recovery_sweep(
    mean_photon_numbers,
    intercept_fractions,
    raw_size: int,
    seeds,
):
    rows = []
    print("mu intercept sifted QBER recovery solve_s status seed")
    for mu in mean_photon_numbers:
        for intercept_fraction in intercept_fractions:
            for seed in seeds:
                row = run_recovery_attack(
                    raw_size,
                    float(mu),
                    float(intercept_fraction),
                    int(seed),
                )
                rows.append(row)
                print(
                    f"{mu:3.1f} {intercept_fraction:8.1%} "
                    f"{row['sifted_size']:6d} {row['measured_qber']:6.2%} "
                    f"{row['recovery_accuracy']:8.2%} "
                    f"{row['solve_seconds']:7.3f} {row['status']} {seed}"
                )
    return rows


def write_recovery_rows(rows, path: Path):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_recovery_rows(path: Path):
    integer_fields = {
        "raw_size",
        "sifted_size",
        "seed",
        "remaining_errors",
        "constraints",
        "rank",
    }
    float_fields = {
        "mean_photon_number",
        "intercept_fraction",
        "realized_intercept_fraction",
        "expected_qber",
        "measured_qber",
        "recovery_accuracy",
        "baseline_accuracy",
        "solve_seconds",
    }
    rows = []
    with path.open(newline="") as source:
        for raw_row in csv.DictReader(source):
            missing = (integer_fields | float_fields | {"optimal", "status"}) - set(
                raw_row
            )
            if missing:
                raise ValueError(
                    f"{path} is not recovery-sweep data; use --recompute"
                )
            row = dict(raw_row)
            for field in integer_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            row["optimal"] = row["optimal"].lower() == "true"
            rows.append(row)
    return rows


def aggregate_recovery(rows):
    mean_photon_numbers = sorted(
        {float(row["mean_photon_number"]) for row in rows}
    )
    intercept_fractions = sorted(
        {float(row["intercept_fraction"]) for row in rows}
    )
    recovery = np.empty(
        (len(intercept_fractions), len(mean_photon_numbers)), dtype=float
    )
    sifted_sizes = []
    for mu_index, mu in enumerate(mean_photon_numbers):
        for intercept_index, fraction in enumerate(intercept_fractions):
            selected = [
                row
                for row in rows
                if np.isclose(row["mean_photon_number"], mu)
                and np.isclose(row["intercept_fraction"], fraction)
            ]
            if not selected:
                raise ValueError("recovery sweep is not a complete grid")
            recovery[intercept_index, mu_index] = np.mean(
                [row["recovery_accuracy"] for row in selected]
            )
            sifted_sizes.extend(row["sifted_size"] for row in selected)
    return (
        np.asarray(mean_photon_numbers),
        np.asarray(intercept_fractions),
        recovery,
        float(np.mean(sifted_sizes)),
    )


def plot_operating_region(rows, qber_limit: float, output_path: Path):
    mean_photon_numbers, intercept_fractions, recovery, mean_sifted = (
        aggregate_recovery(rows)
    )
    mu_grid, intercept_grid = np.meshgrid(
        mean_photon_numbers, intercept_fractions
    )
    full_qber = np.array(
        [srm_probabilities(mu).qber for mu in mean_photon_numbers]
    )
    effective_qber = intercept_grid * full_qber[np.newaxis, :]
    qber_percent = effective_qber * 100.0
    intercept_percent = intercept_grid * 100.0

    recovery_percent = recovery * 100.0
    recovery_norm = colors.Normalize(vmin=50.0, vmax=100.0)
    recovery_colors = cm.viridis(recovery_norm(recovery_percent))
    safe_qber = np.where(effective_qber <= qber_limit, qber_percent, np.nan)
    unsafe_qber = np.where(effective_qber > qber_limit, qber_percent, np.nan)

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        mu_grid,
        intercept_percent,
        safe_qber,
        facecolors=recovery_colors,
        edgecolor="0.35",
        linewidth=0.35,
        alpha=0.9,
        antialiased=True,
        shade=False,
    )
    ax.scatter(
        mu_grid[effective_qber <= qber_limit],
        intercept_percent[effective_qber <= qber_limit],
        qber_percent[effective_qber <= qber_limit],
        c=recovery_percent[effective_qber <= qber_limit],
        cmap=cm.viridis,
        norm=recovery_norm,
        s=24,
        edgecolors="black",
        linewidths=0.25,
        depthshade=False,
    )
    ax.plot_surface(
        mu_grid,
        intercept_percent,
        unsafe_qber,
        color="tab:red",
        edgecolor="0.45",
        linewidth=0.25,
        alpha=0.23,
        antialiased=True,
    )

    limit_plane_x, limit_plane_y = np.meshgrid(
        [mean_photon_numbers[0], mean_photon_numbers[-1]],
        [0.0, 100.0],
    )
    ax.plot_surface(
        limit_plane_x,
        limit_plane_y,
        np.full_like(limit_plane_x, qber_limit * 100.0),
        color="tab:red",
        alpha=0.1,
        shade=False,
    )

    max_intercept = np.array(
        [maximum_intercept_fraction(mu, qber_limit) for mu in mean_photon_numbers]
    )
    boundary_qber = max_intercept * full_qber * 100.0
    ax.plot(
        mean_photon_numbers,
        max_intercept * 100.0,
        boundary_qber,
        color="black",
        linewidth=3.0,
    )
    ax.plot(
        mean_photon_numbers,
        np.full_like(mean_photon_numbers, 100.0),
        full_qber * 100.0,
        color="tab:blue",
        linewidth=2.2,
    )

    ax.set_xlabel("Mean photon number μ = α²", labelpad=12)
    ax.set_ylabel("Eve interception rate (%)", labelpad=12)
    ax.set_zlabel("Expected attack-induced QBER (%)", labelpad=12)
    ax.set_title(
        "Hybrid key recovery under partial 4-PSK interception\n"
        f"mean sifted key size ≈ {mean_sifted / 1000:.1f}k bits",
        pad=18,
    )
    ax.set_xlim(mean_photon_numbers[0], 1.0)
    ax.set_ylim(0.0, 100.0)
    ax.set_zlim(0.0, max(float(np.nanmax(qber_percent)) * 1.03, 12.0))
    ax.view_init(elev=26, azim=-55)

    colorbar = fig.colorbar(
        cm.ScalarMappable(norm=recovery_norm, cmap=cm.viridis),
        ax=ax,
        shrink=0.68,
        pad=0.1,
    )
    colorbar.set_label("Recovered sifted-key bits after hybrid solve (%)")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=3,
            label=f"Max interception at {qber_limit:.0%} QBER",
        ),
        Line2D(
            [0],
            [0],
            color="tab:blue",
            linewidth=2.2,
            label="100% interception",
        ),
        Patch(
            facecolor="tab:red",
            alpha=0.23,
            label=f"Above {qber_limit:.0%} QBER",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.98))

    fig.subplots_adjust(left=0.02, right=0.80, bottom=0.07, top=0.89)
    fig.savefig(output_path, dpi=180)
    return fig


def print_selected_points(qber_limit: float):
    print("mu   HBER(full) QBER(full) max intercept at limit")
    for mu in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        probabilities = srm_probabilities(mu)
        max_fraction = maximum_intercept_fraction(mu, qber_limit)
        print(
            f"{mu:3.1f}  {probabilities.hber:9.2%} "
            f"{probabilities.qber:9.2%} {max_fraction:22.2%}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean-photon-min", type=float, default=0.3)
    parser.add_argument("--mean-photon-max", type=float, default=1.0)
    parser.add_argument("--mean-photon-points", type=int, default=8)
    parser.add_argument("--intercept-points", type=int, default=9)
    parser.add_argument("--raw-size", type=int, default=DEFAULT_RAW_SIZE)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--qber-limit", type=float, default=DEFAULT_QBER_LIMIT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mean_photon_partial_intercept_3d.png"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("mean_photon_partial_intercept.csv"),
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="rerun the bounded hybrid recovery sweep instead of using the CSV",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.mean_photon_min < 0.0:
        parser.error("--mean-photon-min must be non-negative")
    if not args.mean_photon_min < args.mean_photon_max <= 1.0:
        parser.error("--mean-photon-max must be above the minimum and at most 1")
    if args.mean_photon_points < 2 or args.intercept_points < 2:
        parser.error("surface grids require at least two points per axis")
    if args.raw_size <= 0:
        parser.error("--raw-size must be positive")

    if args.recompute or not args.csv.exists():
        mean_photon_numbers = np.linspace(
            args.mean_photon_min,
            args.mean_photon_max,
            args.mean_photon_points,
        )
        intercept_fractions = np.linspace(0.0, 1.0, args.intercept_points)
        rows = run_recovery_sweep(
            mean_photon_numbers,
            intercept_fractions,
            args.raw_size,
            tuple(args.seeds),
        )
        write_recovery_rows(rows, args.csv)
        print(f"Saved recovery data to {args.csv}")
    else:
        rows = read_recovery_rows(args.csv)
        print(f"Loaded recovery data from {args.csv}")

    plot_operating_region(rows, args.qber_limit, args.output)
    print_selected_points(args.qber_limit)
    print(f"Saved plot to {args.output}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
