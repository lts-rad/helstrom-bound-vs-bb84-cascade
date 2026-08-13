"""Reproducible old-vs-new benchmark for CASCADE constraint solving."""

from __future__ import annotations

import argparse
import contextlib
import io
from time import perf_counter

from z3 import Bool, If, Solver, Sum, Xor, is_true, sat

from hybrid_constraint_solver import (
    HybridConstraintSolver,
    count_affine_violations,
    score_result,
    solve_information_set_legacy,
)
from quadrature_attack_model import ERATE, IRATE, QuadratureAttackModel


def _stat(stats, *names):
    available = {key: stats.get_key_value(key) for key in stats.keys()}
    return int(sum(available.get(name, 0) for name in names))


def run_legacy_monolithic(model, constraints, timeout_ms=30_000):
    """Faithful copy of the pre-hybrid Z3 encoding, including oracle counts."""
    n = model.sifted_key_size
    solver = Solver()
    solver.set(timeout=timeout_ms)
    alice = [Bool(f"legacy_alice_{index}") for index in range(n)]

    for public_constraint in constraints:
        indices = public_constraint["indices"]
        parity = public_constraint["alice_parity"]
        if len(indices) == 1:
            solver.add(alice[indices[0]] == bool(parity))
        else:
            expression = alice[indices[0]]
            for index in indices[1:]:
                expression = Xor(expression, alice[index])
            solver.add(expression == bool(parity))

    reliable = [index for index, value in enumerate(model.eve_had_correct_basis) if value]
    unreliable = [index for index, value in enumerate(model.eve_had_correct_basis) if not value]
    if reliable:
        count = Sum([If(alice[i] != bool(model.bob_bits[i]), 1, 0) for i in reliable])
        expected = int(len(reliable) * (ERATE / 3) / (IRATE + ERATE / 3))
        tolerance = max(2, int(len(reliable) * 0.02))
        solver.add(count >= max(0, expected - tolerance), count <= expected + tolerance)
    if unreliable:
        count = Sum([If(alice[i] != bool(model.bob_bits[i]), 1, 0) for i in unreliable])
        expected = int(len(unreliable) * 0.5)
        tolerance = max(3, int(len(unreliable) * 0.15))
        solver.add(count >= max(0, expected - tolerance), count <= expected + tolerance)
    total = Sum([If(alice[i] != bool(model.bob_bits[i]), 1, 0) for i in range(n)])
    expected = int(n * ERATE / 2)
    tolerance = max(5, int(n * 0.01))
    solver.add(total >= expected - tolerance, total <= expected + tolerance)

    started = perf_counter()
    check = solver.check()
    elapsed = perf_counter() - started
    stats = solver.statistics()
    result = {
        "status": str(check),
        "time": elapsed,
        "conflicts": _stat(stats, "conflicts", "sat conflicts"),
        "decisions": _stat(stats, "decisions", "sat decisions"),
        "propagations": _stat(
            stats,
            "propagations",
            "binary propagations",
            "pb propagations",
            "sat propagations 2ary",
            "sat propagations nary",
        ),
        "accuracy": 0.0,
        "violations": None,
    }
    if check == sat:
        z3_model = solver.model()
        solution = tuple(
            int(is_true(z3_model.eval(variable, model_completion=True))) for variable in alice
        )
        result["accuracy"] = sum(
            left == right for left, right in zip(solution, model.alice_bits)
        ) / n
        result["violations"] = count_affine_violations(solution, constraints)
    return result


def benchmark(raw_sizes, seeds, include_slow_legacy=True):
    print(
        "raw seed sift rows solver                    status       accuracy  "
        "viol  time_s  conflicts decisions propagations"
    )
    for raw_size in raw_sizes:
        for seed in seeds:
            # CASCADE is noisy; hide its progress output so rows remain machine-readable.
            with contextlib.redirect_stdout(io.StringIO()):
                model = QuadratureAttackModel(raw_size, seed)
                constraints = model.run_cascade()

            if include_slow_legacy:
                legacy_z3 = run_legacy_monolithic(model, constraints)
                print(
                    f"{raw_size:4d} {seed:4d} {model.sifted_key_size:4d} {len(constraints):4d} "
                    f"legacy-oracle-z3         {legacy_z3['status']:<12} "
                    f"{legacy_z3['accuracy']:8.4f} {str(legacy_z3['violations']):>5} "
                    f"{legacy_z3['time']:7.3f} {legacy_z3['conflicts']:10d} "
                    f"{legacy_z3['decisions']:9d} {legacy_z3['propagations']:12d}"
                )

            started = perf_counter()
            legacy_solution = solve_information_set_legacy(
                model.eve_measurements, model.eve_had_correct_basis, constraints
            )
            legacy_time = perf_counter() - started
            legacy_accuracy = sum(
                left == right for left, right in zip(legacy_solution, model.alice_bits)
            ) / model.sifted_key_size
            legacy_violations = count_affine_violations(legacy_solution, constraints)
            print(
                f"{raw_size:4d} {seed:4d} {model.sifted_key_size:4d} {len(constraints):4d} "
                f"legacy-gf2-heuristic       feasible     {legacy_accuracy:8.4f} "
                f"{legacy_violations:5d} {legacy_time:7.3f} {0:10d} {0:9d} {0:12d}"
            )

            hybrid = HybridConstraintSolver.from_model(model, constraints).solve()
            hybrid_score = score_result(model, hybrid)
            print(
                f"{raw_size:4d} {seed:4d} {model.sifted_key_size:4d} {len(constraints):4d} "
                f"hybrid-affine-ml            {hybrid.status:<12} "
                f"{hybrid_score['accuracy']:8.4f} {hybrid.affine_violations:5d} "
                f"{hybrid.total_seconds:7.3f} {hybrid.conflicts:10d} "
                f"{hybrid.decisions:9d} {hybrid.propagations:12d}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-sizes", type=int, nargs="+", default=[1024])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--fast", action="store_true", help="skip the 30-second legacy Z3 path"
    )
    args = parser.parse_args()
    benchmark(args.raw_sizes, args.seeds, include_slow_legacy=not args.fast)


if __name__ == "__main__":
    main()
