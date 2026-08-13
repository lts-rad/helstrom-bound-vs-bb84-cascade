import itertools
import unittest

from hybrid_constraint_solver import (
    HybridConstraintSolver,
    count_affine_violations,
    solve_information_set_legacy,
)


def constraint(indices, parity):
    return {"indices": list(indices), "alice_parity": parity}


class HybridConstraintSolverUnitTests(unittest.TestCase):
    def test_matches_exhaustive_maximum_likelihood(self):
        measurements = [0, 1, 0, 1, 0, 1]
        reliable = [True, True, True, False, False, True]
        constraints = [
            constraint([0, 1, 3], 0),
            constraint([1, 2, 4], 1),
            constraint([0, 2, 5], 1),
        ]

        result = HybridConstraintSolver(measurements, reliable, constraints).solve()
        feasible = []
        for candidate in itertools.product((0, 1), repeat=len(measurements)):
            if count_affine_violations(candidate, constraints) == 0:
                cost = sum(
                    candidate[index] != measurements[index]
                    for index, is_reliable in enumerate(reliable)
                    if is_reliable
                )
                feasible.append((cost, candidate))

        self.assertTrue(result.optimal)
        self.assertEqual(result.reliable_errors, min(cost for cost, _ in feasible))
        self.assertEqual(result.affine_violations, 0)

    def test_improves_on_gf2_information_set_counterexample(self):
        # The old heuristic sets free e2=0 and forces e0=e1=1 (cost 2).
        # The ML solution sets e2=1 and e0=e1=0 (cost 1).
        measurements = [0, 0, 0]
        reliable = [True, True, True]
        constraints = [constraint([0, 2], 1), constraint([1, 2], 1)]

        legacy = solve_information_set_legacy(measurements, reliable, constraints)
        result = HybridConstraintSolver(measurements, reliable, constraints).solve()

        legacy_cost = sum(left != right for left, right in zip(legacy, measurements))
        self.assertEqual(legacy_cost, 2)
        self.assertEqual(result.reliable_errors, 1)
        self.assertEqual(result.solution, (0, 0, 1))

    def test_duplicate_indices_cancel_over_gf2(self):
        result = HybridConstraintSolver(
            [0, 0],
            [True, False],
            [constraint([0, 0, 1], 1)],
        ).solve()
        self.assertEqual(result.solution[1], 1)
        self.assertEqual(result.affine_violations, 0)

    def test_rejects_inconsistent_public_transcript(self):
        constraints = [constraint([0], 0), constraint([0], 1)]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            HybridConstraintSolver([0], [True], constraints).solve()

    def test_unit_lower_bound_skips_residual_sat(self):
        result = HybridConstraintSolver(
            [0, 0, 0],
            [True, True, True],
            [constraint([0], 1), constraint([1, 2], 0)],
        ).solve()
        self.assertEqual(result.reliable_errors, 1)
        self.assertEqual(result.forced_error_lower_bound, 1)
        self.assertEqual(result.z3_checks, 0)
        self.assertTrue(result.optimal)

    def test_model_factory_does_not_read_simulator_ground_truth(self):
        class AttackerView:
            eve_measurements = [0, 1]
            eve_had_correct_basis = [True, False]

            @property
            def bob_bits(self):
                raise AssertionError("solver read Bob ground truth")

            @property
            def alice_bits(self):
                raise AssertionError("solver read Alice ground truth")

        solver = HybridConstraintSolver.from_model(
            AttackerView(), [constraint([0, 1], 1)]
        )
        self.assertEqual(solver.solve().affine_violations, 0)


class RepresentativeCascadeComparisonTests(unittest.TestCase):
    def test_new_solver_dominates_old_information_set_on_seeded_instances(self):
        from quadrature_attack_model import QuadratureAttackModel

        for seed in (42, 43, 44):
            with self.subTest(seed=seed):
                model = QuadratureAttackModel(1024, seed)
                constraints = model.run_cascade()
                legacy = solve_information_set_legacy(
                    model.eve_measurements,
                    model.eve_had_correct_basis,
                    constraints,
                )
                result = HybridConstraintSolver.from_model(model, constraints).solve()

                self.assertEqual(count_affine_violations(legacy, constraints), 0)
                self.assertEqual(result.affine_violations, 0)
                legacy_cost = sum(
                    legacy[index] != model.eve_measurements[index]
                    for index, reliable in enumerate(model.eve_had_correct_basis)
                    if reliable
                )
                self.assertLessEqual(result.reliable_errors, legacy_cost)


if __name__ == "__main__":
    unittest.main()
