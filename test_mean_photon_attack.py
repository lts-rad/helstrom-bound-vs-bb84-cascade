"""Tests for the §2.1 square-root-measurement attack model."""

import unittest

from hybrid_constraint_solver import HybridConstraintSolver
from plot_3d_mean_photon import (
    expected_attack_rates,
    maximum_intercept_fraction,
)
from qpsk_srm_attack import MeanPhotonAttackModel, srm_probabilities


class SRMProbabilityTests(unittest.TestCase):
    def test_matches_pdf_table(self):
        expected = {
            0.5: (0.738, 0.223, 0.038),
            0.7: (0.824, 0.153, 0.022),
            1.0: (0.908, 0.083, 0.009),
        }
        for mu, (correct, cross, same) in expected.items():
            with self.subTest(mu=mu):
                probabilities = srm_probabilities(mu)
                self.assertAlmostEqual(probabilities.correct, correct, places=3)
                self.assertAlmostEqual(probabilities.cross, cross, places=3)
                self.assertAlmostEqual(probabilities.same, same, places=3)
                self.assertAlmostEqual(
                    probabilities.correct
                    + probabilities.cross
                    + probabilities.same,
                    1.0,
                    places=12,
                )

    def test_qber_uses_dynamic_cross_and_same_basis_terms(self):
        low = srm_probabilities(0.5)
        high = srm_probabilities(1.0)
        self.assertAlmostEqual(low.qber, low.same + low.cross / 2.0)
        self.assertAlmostEqual(high.qber, high.same + high.cross / 2.0)
        self.assertNotAlmostEqual(low.qber / low.hber, high.qber / high.hber)
        self.assertGreater(low.hber, high.hber)
        self.assertGreater(low.qber, high.qber)


class MeanPhotonAttackModelTests(unittest.TestCase):
    def test_cascade_and_hybrid_use_dynamic_model(self):
        model = MeanPhotonAttackModel(1024, seed=42, mean_photon_number=0.7)
        constraints = model.run_cascade()
        result = HybridConstraintSolver.from_model(model, constraints).solve()
        self.assertEqual(result.affine_violations, 0)
        self.assertTrue(result.optimal)
        self.assertAlmostEqual(model.expected_qber(), model.srm.qber)

    def test_realized_rates_track_expected_distribution(self):
        model = MeanPhotonAttackModel(100_000, seed=42, mean_photon_number=0.8)
        realized = model.realized_outcome_rates()
        self.assertAlmostEqual(realized["correct"], model.srm.correct, delta=0.01)
        self.assertAlmostEqual(realized["cross"], model.srm.cross, delta=0.01)
        self.assertAlmostEqual(realized["same"], model.srm.same, delta=0.01)
        self.assertAlmostEqual(model.calculate_hber(), model.srm.hber, delta=0.01)
        self.assertAlmostEqual(model.calculate_qber(), model.srm.qber, delta=0.01)


class PartialInterceptOperatingRegionTests(unittest.TestCase):
    def test_attack_qber_and_hber_scale_with_intercept_fraction(self):
        full = expected_attack_rates(0.5, 1.0)
        partial = expected_attack_rates(0.5, 0.4)
        self.assertAlmostEqual(
            partial["qber_effective"], 0.4 * full["qber_effective"]
        )
        self.assertAlmostEqual(
            partial["hber_effective"], 0.4 * full["hber_effective"]
        )

    def test_maximum_intercept_obeys_eleven_percent_limit(self):
        low_mu_fraction = maximum_intercept_fraction(0.5, 0.11)
        high_mu_fraction = maximum_intercept_fraction(1.0, 0.11)
        self.assertLess(low_mu_fraction, 1.0)
        self.assertAlmostEqual(
            expected_attack_rates(0.5, low_mu_fraction)["qber_effective"],
            0.11,
        )
        self.assertEqual(high_mu_fraction, 1.0)
        self.assertLess(
            expected_attack_rates(1.0, high_mu_fraction)["qber_effective"],
            0.11,
        )


if __name__ == "__main__":
    unittest.main()
