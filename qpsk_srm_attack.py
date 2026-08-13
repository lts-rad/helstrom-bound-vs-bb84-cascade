"""4-PSK square-root-measurement attack with variable mean photon number."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from quadrature_attack_model import QuadratureAttackModel
from cascade.key import Key
from cascade.shuffle import Shuffle


@dataclass(frozen=True)
class SRMProbabilities:
    """Outcome probabilities from Equations 6-8 of 4pskbb84.pdf §2.1."""

    correct: float
    cross: float
    same: float

    @property
    def hber(self) -> float:
        """Eve's total state-discrimination error probability."""
        return self.cross + self.same

    @property
    def qber(self) -> float:
        """Alice-Bob error after resend and basis sifting."""
        return self.same + self.cross / 2.0


def srm_probabilities(mean_photon_number: float) -> SRMProbabilities:
    """Return correct, cross-basis, and same-basis-wrong SRM probabilities."""
    if mean_photon_number < 0.0:
        raise ValueError("mean_photon_number must be non-negative")

    mu = float(mean_photon_number)
    decay = np.exp(-mu)
    h1 = 2.0 * decay * (np.cosh(mu) + np.cos(mu))
    h2 = 2.0 * decay * (np.sinh(mu) + np.sin(mu))
    h3 = 2.0 * decay * (np.cosh(mu) - np.cos(mu))
    h4 = 2.0 * decay * (np.sinh(mu) - np.sin(mu))
    roots = np.sqrt(np.maximum((h1, h2, h3, h4), 0.0))
    s1, s2, s3, s4 = roots

    correct = (s1 + s2 + s3 + s4) ** 2 / 16.0
    cross = ((s1 - s3) ** 2 + (s2 - s4) ** 2) / 8.0
    same = (s1 - s2 + s3 - s4) ** 2 / 16.0

    # Floating-point cancellation is most visible near mu=0. Normalize the
    # three physical outcomes while preserving their relative weights.
    total = correct + cross + same
    return SRMProbabilities(
        correct=float(correct / total),
        cross=float(cross / total),
        same=float(same / total),
    )


class MeanPhotonAttackModel(QuadratureAttackModel):
    """Intercept-resend simulation using the full SRM outcome distribution."""

    def __init__(self, raw_key_size: int, seed: int, mean_photon_number: float):
        self.raw_key_size = raw_key_size
        self.mean_photon_number = float(mean_photon_number)
        self.srm = srm_probabilities(mean_photon_number)

        random.seed(seed)
        np.random.seed(seed)
        Key.set_random_seed(seed)
        Shuffle.set_random_seed(seed + 1)

        self.alice_bits_raw = [random.randint(0, 1) for _ in range(raw_key_size)]
        self.alice_bases = [random.randint(0, 1) for _ in range(raw_key_size)]
        self.bob_bases = [random.randint(0, 1) for _ in range(raw_key_size)]

        self.eve_measurements_raw = []
        self.eve_had_correct_basis_raw = []
        self.eve_outcome_class_raw = []
        self.bob_bits_raw = []

        correct_cutoff = self.srm.correct
        same_cutoff = correct_cutoff + self.srm.same
        for alice_bit in self.alice_bits_raw:
            outcome_draw = random.random()
            cross_eve_bit = random.randint(0, 1)
            cross_bob_bit = random.randint(0, 1)

            if outcome_draw < correct_cutoff:
                outcome_class = "correct"
                eve_bit = alice_bit
                correct_basis = True
            elif outcome_draw < same_cutoff:
                outcome_class = "same"
                eve_bit = 1 - alice_bit
                correct_basis = True
            else:
                outcome_class = "cross"
                eve_bit = cross_eve_bit
                correct_basis = False

            self.eve_outcome_class_raw.append(outcome_class)
            self.eve_measurements_raw.append(eve_bit)
            self.eve_had_correct_basis_raw.append(correct_basis)
            self.bob_bits_raw.append(eve_bit if correct_basis else cross_bob_bit)

        self.perform_sifting()
        self.eve_outcome_class = [
            self.eve_outcome_class_raw[index] for index in self.sifted_indices
        ]

    def expected_hber(self) -> float:
        return self.srm.hber

    def expected_qber(self) -> float:
        return self.srm.qber

    def calculate_hber(self) -> float:
        if not self.eve_outcome_class:
            return 0.0
        errors = sum(value != "correct" for value in self.eve_outcome_class)
        return errors / len(self.eve_outcome_class)

    def realized_outcome_rates(self) -> dict[str, float]:
        total = len(self.eve_outcome_class)
        if total == 0:
            return {"correct": 0.0, "cross": 0.0, "same": 0.0}
        return {
            name: self.eve_outcome_class.count(name) / total
            for name in ("correct", "cross", "same")
        }
