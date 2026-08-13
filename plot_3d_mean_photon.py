"""Plot the 4-PSK partial-interception operating region from PDF §2.1."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from qpsk_srm_attack import srm_probabilities


DEFAULT_QBER_LIMIT = 0.11


def expected_attack_rates(
    mean_photon_number: float, intercept_fraction: float
) -> dict[str, float]:
    """Expected attack rates with unsampled signals passed through unchanged."""
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


def build_surface(mean_photon_numbers, intercept_fractions):
    mu_grid, intercept_grid = np.meshgrid(
        mean_photon_numbers, intercept_fractions
    )
    full_hber = np.array(
        [srm_probabilities(mu).hber for mu in mean_photon_numbers]
    )
    full_qber = np.array(
        [srm_probabilities(mu).qber for mu in mean_photon_numbers]
    )
    effective_hber = intercept_grid * full_hber[np.newaxis, :]
    effective_qber = intercept_grid * full_qber[np.newaxis, :]
    return mu_grid, intercept_grid, effective_hber, effective_qber


def plot_operating_region(
    mean_photon_numbers,
    intercept_fractions,
    qber_limit: float,
    output_path: Path,
):
    mu_grid, intercept_grid, effective_hber, effective_qber = build_surface(
        mean_photon_numbers, intercept_fractions
    )
    qber_percent = effective_qber * 100.0
    intercept_percent = intercept_grid * 100.0

    hber_norm = colors.Normalize(
        vmin=float(np.min(effective_hber) * 100.0),
        vmax=float(np.max(effective_hber) * 100.0),
    )
    hber_colors = cm.viridis(hber_norm(effective_hber * 100.0))
    safe_qber = np.where(effective_qber <= qber_limit, qber_percent, np.nan)
    unsafe_qber = np.where(effective_qber > qber_limit, qber_percent, np.nan)

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        mu_grid,
        intercept_percent,
        safe_qber,
        facecolors=hber_colors,
        edgecolor="0.35",
        linewidth=0.25,
        alpha=0.88,
        antialiased=True,
        shade=False,
    )
    ax.plot_surface(
        mu_grid,
        intercept_percent,
        unsafe_qber,
        color="tab:red",
        edgecolor="0.45",
        linewidth=0.2,
        alpha=0.22,
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
        alpha=0.11,
        shade=False,
    )

    max_intercept = np.array(
        [maximum_intercept_fraction(mu, qber_limit) for mu in mean_photon_numbers]
    )
    boundary_qber = np.array(
        [
            expected_attack_rates(mu, fraction)["qber_effective"] * 100.0
            for mu, fraction in zip(mean_photon_numbers, max_intercept)
        ]
    )
    ax.plot(
        mean_photon_numbers,
        max_intercept * 100.0,
        boundary_qber,
        color="black",
        linewidth=3.0,
        label="Maximum interception at QBER limit",
    )

    full_qber = np.array(
        [srm_probabilities(mu).qber * 100.0 for mu in mean_photon_numbers]
    )
    ax.plot(
        mean_photon_numbers,
        np.full_like(mean_photon_numbers, 100.0),
        full_qber,
        color="tab:blue",
        linewidth=2.2,
        label="Full-interception QBER",
    )

    ax.set_xlabel("Mean photon number μ = α²", labelpad=12)
    ax.set_ylabel("Eve interception rate (%)", labelpad=12)
    ax.set_zlabel("Expected attack-induced QBER (%)", labelpad=12)
    ax.set_title(
        "4-PSK partial interception under a QBER acceptance limit", pad=18
    )
    ax.set_xlim(mean_photon_numbers[0], mean_photon_numbers[-1])
    ax.set_ylim(0.0, 100.0)
    ax.set_zlim(0.0, max(float(np.nanmax(qber_percent)) * 1.03, 12.0))
    ax.view_init(elev=26, azim=-55)

    colorbar = fig.colorbar(
        cm.ScalarMappable(norm=hber_norm, cmap=cm.viridis),
        ax=ax,
        shrink=0.68,
        pad=0.1,
    )
    colorbar.set_label("Effective HBER = intercepted fraction × HBER (%)")

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
            alpha=0.22,
            label=f"Above {qber_limit:.0%} QBER",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.98))

    fig.subplots_adjust(left=0.02, right=0.80, bottom=0.07, top=0.91)
    fig.savefig(output_path, dpi=180)
    return fig


def write_summary(mean_photon_numbers, qber_limit: float, path: Path):
    fieldnames = [
        "mean_photon_number",
        "p_correct",
        "p_cross",
        "p_same",
        "hber_full_intercept",
        "qber_full_intercept",
        "max_intercept_fraction",
        "qber_at_max_intercept",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for mu in mean_photon_numbers:
            probabilities = srm_probabilities(mu)
            max_fraction = maximum_intercept_fraction(mu, qber_limit)
            writer.writerow(
                {
                    "mean_photon_number": mu,
                    "p_correct": probabilities.correct,
                    "p_cross": probabilities.cross,
                    "p_same": probabilities.same,
                    "hber_full_intercept": probabilities.hber,
                    "qber_full_intercept": probabilities.qber,
                    "max_intercept_fraction": max_fraction,
                    "qber_at_max_intercept": max_fraction
                    * probabilities.qber,
                }
            )


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
    parser.add_argument("--mean-photon-max", type=float, default=1.2)
    parser.add_argument("--mean-photon-points", type=int, default=91)
    parser.add_argument("--intercept-points", type=int, default=81)
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
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.mean_photon_min < 0.0:
        parser.error("--mean-photon-min must be non-negative")
    if args.mean_photon_max <= args.mean_photon_min:
        parser.error("--mean-photon-max must exceed --mean-photon-min")
    if args.mean_photon_points < 2 or args.intercept_points < 2:
        parser.error("surface grids require at least two points per axis")

    mean_photon_numbers = np.linspace(
        args.mean_photon_min,
        args.mean_photon_max,
        args.mean_photon_points,
    )
    intercept_fractions = np.linspace(0.0, 1.0, args.intercept_points)
    write_summary(mean_photon_numbers, args.qber_limit, args.csv)
    plot_operating_region(
        mean_photon_numbers,
        intercept_fractions,
        args.qber_limit,
        args.output,
    )
    print_selected_points(args.qber_limit)
    print(f"Saved plot to {args.output}")
    print(f"Saved data to {args.csv}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
