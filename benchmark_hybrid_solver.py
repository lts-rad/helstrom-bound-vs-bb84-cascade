"""Reproducible benchmark for the active hybrid CASCADE solver."""

from __future__ import annotations

import argparse
import contextlib
import io

from hybrid_constraint_solver import HybridConstraintSolver, score_result
from quadrature_attack_model import QuadratureAttackModel


def benchmark(raw_sizes, seeds):
    print(
        "raw seed sift rows status       accuracy viol time_s checks "
        "conflicts decisions propagations"
    )
    for raw_size in raw_sizes:
        for seed in seeds:
            # CASCADE is noisy; hide its progress output so rows remain readable.
            with contextlib.redirect_stdout(io.StringIO()):
                model = QuadratureAttackModel(raw_size, seed)
                constraints = model.run_cascade()

            result = HybridConstraintSolver.from_model(model, constraints).solve()
            score = score_result(model, result)
            print(
                f"{raw_size:6d} {seed:4d} {model.sifted_key_size:6d} "
                f"{len(constraints):6d} {result.status:<12} "
                f"{score['accuracy']:8.4f} {result.affine_violations:4d} "
                f"{result.total_seconds:7.3f} {result.z3_checks:6d} "
                f"{result.conflicts:9d} {result.decisions:9d} "
                f"{result.propagations:12d}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-sizes", type=int, nargs="+", default=[1024])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    benchmark(args.raw_sizes, args.seeds)


if __name__ == "__main__":
    main()
