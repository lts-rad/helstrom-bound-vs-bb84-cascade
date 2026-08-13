"""Tests for the hybrid partial-interception sweep."""

import unittest

from plot_3d_partial_sampling import PartialSamplingModel, run_attack
from quadrature_attack_model import QuadratureAttackModel


class PartialSamplingTests(unittest.TestCase):
    def test_zero_sampling_leaves_bob_undisturbed_and_eve_uninformed(self):
        model = PartialSamplingModel(512, seed=42, sampling_rate=0.0)
        self.assertEqual(model.alice_bits, model.bob_bits)
        self.assertEqual(model.calculate_qber(), 0.0)
        self.assertFalse(any(model.eve_sampled_sifted))
        self.assertFalse(any(model.eve_had_correct_basis))

    def test_full_sampling_matches_the_full_intercept_model(self):
        expected = QuadratureAttackModel(512, seed=42)
        model = PartialSamplingModel(512, seed=42, sampling_rate=1.0)
        self.assertEqual(model.alice_bits, expected.alice_bits)
        self.assertEqual(model.bob_bits, expected.bob_bits)
        self.assertEqual(model.eve_measurements, expected.eve_measurements)
        self.assertEqual(
            model.eve_had_correct_basis, expected.eve_had_correct_basis
        )
        self.assertTrue(all(model.eve_sampled_sifted))

    def test_partial_attack_uses_feasible_hybrid_solution(self):
        row = run_attack(512, sampling_rate=0.5, seed=42)
        self.assertEqual(row["violations"], 0)
        self.assertTrue(row["optimal"])
        self.assertGreater(row["realized_sampling_rate"], 0.0)
        self.assertLess(row["realized_sampling_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
