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


class PartialInterceptMeanPhotonAttackModel(MeanPhotonAttackModel):
    """Variable-μ attack in which Eve intercepts only a signal fraction."""

    def __init__(
        self,
        raw_key_size: int,
        seed: int,
        mean_photon_number: float,
        intercept_fraction: float,
    ):
        if not 0.0 <= intercept_fraction <= 1.0:
            raise ValueError("intercept_fraction must be between 0 and 1")

        super().__init__(raw_key_size, seed, mean_photon_number)
        self.intercept_fraction = float(intercept_fraction)

        if intercept_fraction == 0.0:
            intercepted_raw = np.zeros(raw_key_size, dtype=bool)
        elif intercept_fraction == 1.0:
            intercepted_raw = np.ones(raw_key_size, dtype=bool)
        else:
            # Keep the underlying SRM outcome realization fixed when comparing
            # interception fractions for the same seed and mean photon number.
            rng = np.random.default_rng(seed + 1_000_003)
            intercepted_raw = rng.random(raw_key_size) < intercept_fraction

        self.eve_intercepted_raw = intercepted_raw.tolist()
        self.eve_intercepted_sifted = []
        for sifted_index, raw_index in enumerate(self.sifted_indices):
            intercepted = bool(intercepted_raw[raw_index])
            self.eve_intercepted_sifted.append(intercepted)
            if intercepted:
                continue

            # Unsampled signals reach Bob unchanged. Eve knows only that she
            # did not measure the position, so it is marked unreliable.
            alice_bit = self.alice_bits[sifted_index]
            self.bob_bits[sifted_index] = alice_bit
            self.bob_bits_raw[raw_index] = alice_bit
            self.eve_measurements[sifted_index] = 0
            self.eve_measurements_raw[raw_index] = 0
            self.eve_had_correct_basis[sifted_index] = False
            self.eve_had_correct_basis_raw[raw_index] = False
            self.eve_outcome_class[sifted_index] = "unintercepted"
            self.eve_outcome_class_raw[raw_index] = "unintercepted"

        self.bob_key._bits = self.bob_bits.copy()
        self.realized_sifted_intercept_fraction = (
            sum(self.eve_intercepted_sifted) / self.sifted_key_size
            if self.sifted_key_size
            else 0.0
        )

    def expected_hber(self) -> float:
        return self.intercept_fraction * self.srm.hber

    def expected_qber(self) -> float:
        return self.intercept_fraction * self.srm.qber

    def calculate_hber(self) -> float:
        if not self.eve_outcome_class:
            return 0.0
        errors = sum(
            value in ("cross", "same") for value in self.eve_outcome_class
        )
        return errors / len(self.eve_outcome_class)
