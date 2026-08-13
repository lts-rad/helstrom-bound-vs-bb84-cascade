"""Hybrid affine/Boolean decoder for the BB84 CASCADE transcript.

Note: this was generated with GLM 5.2 to improve the solver previously in the `old` branch.

The public CASCADE parity answers are exact affine equations over GF(2).  Eve's
measurement quality is not another GF(2) equation: it is a likelihood model.
This module first eliminates the affine core, projects out the positions whose
measurements are uninformative, and then asks Z3 to solve only the residual
minimum-cardinality problem on reliable measurement errors.

Alice's and Bob's simulator-only bit arrays are used only by ``score_result``.
They are deliberately not read by ``HybridConstraintSolver.solve``.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterator, Mapping, Sequence

import numpy as np
from z3 import Bool, BoolVal, PbLe, Solver, Xor, is_true, sat, unsat

Constraint = Mapping[str, object]
Statistics = Mapping[str, int | float]


def _parity(mask: int) -> int:
    return mask.bit_count() & 1


def _mask_from_bits(bits: Sequence[int]) -> int:
    mask = 0
    for index, bit in enumerate(bits):
        if bit & 1:
            mask |= 1 << index
    return mask


def _bit_indices(mask: int) -> Iterator[int]:
    """Yield set-bit indices from least to most significant."""
    while mask:
        low_bit = mask & -mask
        yield low_bit.bit_length() - 1
        mask ^= low_bit


def _xor_equals(variables, rhs: int):
    """Build an XOR equality without the left-deep chains used previously."""
    if not variables:
        return BoolVal(rhs == 0)
    if len(variables) == 1:
        return variables[0] == bool(rhs)
    # z3py's Xor is binary/ternary.  A balanced tree avoids a deep AST.
    layer = list(variables)
    while len(layer) > 1:
        next_layer = []
        for offset in range(0, len(layer), 2):
            if offset + 1 == len(layer):
                next_layer.append(layer[offset])
            else:
                next_layer.append(Xor(layer[offset], layer[offset + 1]))
        layer = next_layer
    return layer[0] == bool(rhs)


def _z3_statistics(solver: Solver) -> dict[str, int | float]:
    stats = solver.statistics()
    result: dict[str, int | float] = {}
    for key in stats.keys():
        value = stats.get_key_value(key)
        if isinstance(value, (int, float)):
            result[key] = value
    return result


@dataclass(frozen=True)
class AffineRow:
    mask: int
    rhs: int

    @property
    def width(self) -> int:
        return self.mask.bit_count()


@dataclass(frozen=True)
class AffineReduction:
    n_variables: int
    rows: tuple[AffineRow, ...]
    pivot_columns: tuple[int, ...]
    rank: int
    inconsistent: bool
    input_rows: int

    @property
    def dependent_rows(self) -> int:
        return self.input_rows - self.rank

    def particular_error_mask(self, fixed_mask: int = 0) -> int:
        """Extend fixed free-variable values through the RREF pivot rows."""
        solution = fixed_mask
        for row, pivot in zip(self.rows, self.pivot_columns):
            without_pivot = row.mask & ~(1 << pivot)
            value = row.rhs ^ _parity(without_pivot & solution)
            if value:
                solution |= 1 << pivot
            else:
                solution &= ~(1 << pivot)
        return solution


@dataclass(frozen=True)
class HybridResult:
    solution: tuple[int, ...]
    status: str
    optimal: bool
    rank: int
    dependent_rows: int
    residual_dimension: int
    projected_equations: int
    projected_variables: int
    reliable_errors: int
    forced_error_lower_bound: int
    affine_violations: int
    z3_checks: int
    conflicts: int
    decisions: int
    propagations: int
    preprocessing_seconds: float
    search_seconds: float
    total_seconds: float
    statistics: Statistics


@dataclass(frozen=True)
class _ResidualResult:
    error_mask: int
    weight: int
    checks: int
    status: str
    optimal: bool
    statistics: Statistics


def reduce_affine_system(
    n_variables: int,
    constraints: Sequence[Constraint],
    *,
    base_bits: Sequence[int] | None = None,
    column_order: Sequence[int] | None = None,
) -> AffineReduction:
    """Bit-packed, vectorized Gauss-Jordan reduction over GF(2).

    The optional base changes variables from Alice bits x to error bits
    e = x XOR base. Rows are packed into NumPy uint64 words; each pivot
    eliminates a column from all target rows in one native vectorized XOR.
    """
    if n_variables < 0:
        raise ValueError("n_variables must be non-negative")
    if base_bits is None:
        base_mask = 0
    else:
        if len(base_bits) != n_variables:
            raise ValueError("base_bits length does not match n_variables")
        base_mask = _mask_from_bits(base_bits)

    if column_order is None:
        order = list(range(n_variables))
    else:
        order = list(column_order)
        if len(order) != n_variables or set(order) != set(range(n_variables)):
            raise ValueError("column_order must be a permutation of all variables")

    row_count = len(constraints)
    word_count = (n_variables + 63) // 64
    matrix = np.zeros((row_count, word_count), dtype=np.uint64)
    rhs = np.zeros(row_count, dtype=np.uint8)

    byte_count = word_count * 8
    for row_index, constraint in enumerate(constraints):
        mask = 0
        for raw_index in constraint["indices"]:
            index = int(raw_index)
            if index < 0 or index >= n_variables:
                raise IndexError(
                    f"constraint index {index} outside 0..{n_variables - 1}"
                )
            mask ^= 1 << index  # duplicate indices cancel over GF(2)
        rhs[row_index] = (
            int(constraint["alice_parity"]) ^ _parity(mask & base_mask)
        ) & 1
        if word_count:
            packed = np.frombuffer(mask.to_bytes(byte_count, "little"), dtype="<u8")
            matrix[row_index] = packed

    row_used = np.zeros(row_count, dtype=bool)
    pivot_list: list[tuple[int, int]] = []

    for column in order:
        word = column >> 6
        shift = np.uint64(column & 63)
        column_bits = (matrix[:, word] >> shift) & np.uint64(1)
        candidates = np.flatnonzero((column_bits == 1) & ~row_used)
        if candidates.size == 0:
            continue

        pivot_row = int(candidates[0])
        row_used[pivot_row] = True
        pivot_list.append((column, pivot_row))

        targets = np.flatnonzero(column_bits == 1)
        targets = targets[targets != pivot_row]
        if targets.size:
            matrix[targets] ^= matrix[pivot_row]
            rhs[targets] ^= rhs[pivot_row]

    zero_rows = ~matrix.any(axis=1) if word_count else np.ones(row_count, dtype=bool)
    inconsistent = bool(np.any(zero_rows & (rhs == 1)))

    reduced_rows = []
    pivot_columns = []
    for column, row_index in pivot_list:
        # Force little-endian words before reconstructing the Python integer
        # mask. This conversion occurs once per independent row.
        row_bytes = matrix[row_index].astype("<u8", copy=False).tobytes()
        reduced_rows.append(
            AffineRow(int.from_bytes(row_bytes, "little"), int(rhs[row_index]))
        )
        pivot_columns.append(column)

    return AffineReduction(
        n_variables=n_variables,
        rows=tuple(reduced_rows),
        pivot_columns=tuple(pivot_columns),
        rank=len(pivot_list),
        inconsistent=inconsistent,
        input_rows=row_count,
    )


def count_affine_violations(
    solution: Sequence[int], constraints: Sequence[Constraint]
) -> int:
    violations = 0
    for constraint in constraints:
        value = 0
        for raw_index in constraint["indices"]:
            value ^= int(solution[int(raw_index)]) & 1
        if value != (int(constraint["alice_parity"]) & 1):
            violations += 1
    return violations


def score_result(model, result: HybridResult) -> dict[str, float | int]:
    """Evaluation-only scoring.  Never called by the solving path."""
    alice_bits = model.alice_bits
    correct = sum(a == b for a, b in zip(alice_bits, result.solution))
    baseline_correct = sum(
        a == e for a, e in zip(alice_bits, model.eve_measurements)
    )
    n = len(alice_bits)
    return {
        "accuracy": correct / n if n else 1.0,
        "baseline_accuracy": baseline_correct / n if n else 1.0,
        "remaining_errors": n - correct,
    }


def _project_reliable_rows(
    reduction: AffineReduction,
    reliable_positions: Sequence[bool],
    reliable_mask: int,
) -> tuple[list[AffineRow], list[int]]:
    """Project the RREF system onto reliable error variables."""
    projected_rows = []
    projected_mask = 0

    for row, pivot in zip(reduction.rows, reduction.pivot_columns):
        if not reliable_positions[pivot]:
            continue
        if row.mask & ~reliable_mask:
            raise AssertionError(
                "projected row still contains an unreliable variable"
            )
        projected_rows.append(row)
        projected_mask |= row.mask

    return projected_rows, list(_bit_indices(projected_mask))


def _forced_error_lower_bound(rows: Sequence[AffineRow]) -> int:
    """Count distinct reliable errors forced by independent unit rows."""
    forced_one_mask = 0
    for row in rows:
        if row.rhs and row.width == 1:
            forced_one_mask |= row.mask
    return forced_one_mask.bit_count()


def _optimize_residual(
    *,
    reduction: AffineReduction,
    projected_rows: Sequence[AffineRow],
    projected_columns: Sequence[int],
    initial_error_mask: int,
    initial_weight: int,
    lower_bound: int,
    timeout_ms: int,
) -> _ResidualResult:
    """Minimize reliable errors, retaining learned clauses between bounds."""
    if lower_bound >= initial_weight:
        return _ResidualResult(
            error_mask=initial_error_mask,
            weight=initial_weight,
            checks=0,
            status="optimal",
            optimal=True,
            statistics={},
        )

    solver = Solver()
    solver.set(timeout=timeout_ms)
    variables = {
        index: Bool(f"reliable_error_{index}") for index in projected_columns
    }

    for row in projected_rows:
        row_variables = [variables[index] for index in _bit_indices(row.mask)]
        solver.add(_xor_equals(row_variables, row.rhs))

    for index, variable in variables.items():
        solver.set_initial_value(
            variable, bool((initial_error_mask >> index) & 1)
        )

    error_mask = initial_error_mask
    weight = initial_weight
    checks = 0
    status = "optimal"
    optimal = True

    while weight > lower_bound and variables:
        solver.add(
            PbLe(
                [(variable, 1) for variable in variables.values()],
                weight - 1,
            )
        )
        checks += 1
        check_result = solver.check()
        if check_result == unsat:
            break
        if check_result != sat:
            status = "feasible-timeout"
            optimal = False
            break

        z3_model = solver.model()
        reliable_error_mask = 0
        for index, variable in variables.items():
            if is_true(z3_model.eval(variable, model_completion=True)):
                reliable_error_mask |= 1 << index

        new_weight = reliable_error_mask.bit_count()
        if new_weight >= weight:
            raise AssertionError(
                "strict cardinality bound did not improve the model"
            )
        weight = new_weight
        error_mask = reduction.particular_error_mask(reliable_error_mask)

    return _ResidualResult(
        error_mask=error_mask,
        weight=weight,
        checks=checks,
        status=status,
        optimal=optimal,
        statistics=_z3_statistics(solver),
    )


def _solver_activity(statistics: Statistics) -> tuple[int, int, int]:
    """Normalize Z3's solver-dependent activity counter names."""
    conflicts = int(
        statistics.get("conflicts", statistics.get("sat conflicts", 0))
    )
    decisions = int(
        statistics.get("decisions", statistics.get("sat decisions", 0))
    )
    propagations = int(
        statistics.get("propagations", 0)
        + statistics.get("binary propagations", 0)
        + statistics.get("pb propagations", 0)
        + statistics.get("sat propagations 2ary", 0)
        + statistics.get("sat propagations nary", 0)
    )
    return conflicts, decisions, propagations


class HybridConstraintSolver:
    """Maximum-likelihood decoder for exact CASCADE parity leakage.

    The solver is intentionally constructed only from Eve-visible inputs:
    ``eve_measurements``, ``eve_had_correct_basis``, and public parity rows.
    It does not retain the simulation model object.
    """

    def __init__(
        self,
        eve_measurements: Sequence[int],
        reliable_positions: Sequence[bool],
        constraints: Sequence[Constraint],
        *,
        timeout_ms: int = 30_000,
    ):
        if len(eve_measurements) != len(reliable_positions):
            raise ValueError(
                "measurement and reliability arrays must have equal length"
            )
        self.eve_measurements = tuple(int(bit) & 1 for bit in eve_measurements)
        self.reliable_positions = tuple(
            bool(value) for value in reliable_positions
        )
        # Copy only the public affine transcript fields.
        self.constraints = tuple(
            {
                "indices": tuple(int(index) for index in constraint["indices"]),
                "alice_parity": int(constraint["alice_parity"]) & 1,
            }
            for constraint in constraints
        )
        self.timeout_ms = timeout_ms

    @classmethod
    def from_model(cls, model, constraints, *, timeout_ms: int = 30_000):
        """Snapshot only attacker-visible fields from a simulation model."""
        return cls(
            model.eve_measurements,
            model.eve_had_correct_basis,
            constraints,
            timeout_ms=timeout_ms,
        )

    def solve(self) -> HybridResult:
        started = perf_counter()
        n_variables = len(self.eve_measurements)
        unreliable = [
            index
            for index, reliable in enumerate(self.reliable_positions)
            if not reliable
        ]
        reliable = [
            index
            for index, is_reliable in enumerate(self.reliable_positions)
            if is_reliable
        ]
        reliable_mask = sum(1 << index for index in reliable)

        reduction = reduce_affine_system(
            n_variables,
            self.constraints,
            base_bits=self.eve_measurements,
            column_order=unreliable + reliable,
        )
        if reduction.inconsistent:
            raise ValueError("public CASCADE parity transcript is inconsistent")

        projected_rows, projected_columns = _project_reliable_rows(
            reduction,
            self.reliable_positions,
            reliable_mask,
        )

        # The RREF particular assignment is a valid upper bound: all free errors
        # are zero and the pivots are determined. Prove or improve that bound.
        candidate_error_mask = reduction.particular_error_mask()
        candidate_weight = (candidate_error_mask & reliable_mask).bit_count()

        forced_lower_bound = _forced_error_lower_bound(projected_rows)
        preprocessing_finished = perf_counter()

        residual = _optimize_residual(
            reduction=reduction,
            projected_rows=projected_rows,
            projected_columns=projected_columns,
            initial_error_mask=candidate_error_mask,
            initial_weight=candidate_weight,
            lower_bound=forced_lower_bound,
            timeout_ms=self.timeout_ms,
        )

        search_finished = perf_counter()
        base_mask = _mask_from_bits(self.eve_measurements)
        solution_mask = base_mask ^ residual.error_mask
        solution = tuple(
            (solution_mask >> index) & 1 for index in range(n_variables)
        )
        violations = count_affine_violations(solution, self.constraints)
        if violations:
            raise AssertionError(
                f"hybrid solution violates {violations} affine constraints"
            )

        conflicts, decisions, propagations = _solver_activity(
            residual.statistics
        )

        return HybridResult(
            solution=solution,
            status=residual.status,
            optimal=residual.optimal,
            rank=reduction.rank,
            dependent_rows=reduction.dependent_rows,
            residual_dimension=n_variables - reduction.rank,
            projected_equations=len(projected_rows),
            projected_variables=len(projected_columns),
            reliable_errors=residual.weight,
            forced_error_lower_bound=forced_lower_bound,
            affine_violations=violations,
            z3_checks=residual.checks,
            conflicts=conflicts,
            decisions=decisions,
            propagations=propagations,
            preprocessing_seconds=preprocessing_finished - started,
            search_seconds=search_finished - preprocessing_finished,
            total_seconds=search_finished - started,
            statistics=residual.statistics,
        )
