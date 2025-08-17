"""
BB84 model based on IDEA.md specifications.

IRATE = 0.908 (90.8% correct basis and value)
ERATE = 0.092 (9.2% error rate, split into 3 equal cases)
Expected QBER = 2*ERATE/3 = 6.134%

Key insights:
1. When Eve has correct basis: Eve's bit = Bob's bit (100% correlation)
2. When Eve has wrong basis: Bob's bit is random (50% match with Alice)
3. CASCADE leaks parity information that helps Eve solve for Alice's bits
"""

import numpy as np
import random
import time
from z3 import *
from cascade.key import Key
from cascade.mock_classical_channel import MockClassicalChannel
from cascade.reconciliation import Reconciliation
from cascade.shuffle import Shuffle

#IRATE = 0.907579  # Rate of correct basis AND correct value (Helstrom bound for α²=1.0)
ERATE = 0.092421  # Error rate from 4-PSK Helstrom bound
IRATE = 1 - ERATE


class QuadratureAttackModel:
    """BB84 with quadrature measurement model from IDEA.md"""

    def __init__(self, raw_key_size=2048, seed=42):
        """
        Initialize the BB84 protocol.

        Args:
            raw_key_size: Size before basis sifting
            seed: Random seed for reproducibility
        """
        self.raw_key_size = raw_key_size

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        Key.set_random_seed(seed)
        Shuffle.set_random_seed(seed + 1)

        # Generate Alice's raw bits and bases
        self.alice_bits_raw = [random.randint(0, 1) for _ in range(raw_key_size)]
        self.alice_bases = [random.randint(0, 1) for _ in range(raw_key_size)]
        self.bob_bases = [random.randint(0, 1) for _ in range(raw_key_size)]

        # Eve's measurement results
        self.eve_measurements_raw = []
        self.eve_had_correct_basis_raw = []

        # Simulate Eve's quadrature measurement according to IDEA.md
        for i in range(raw_key_size):
            r = random.random()

            if r < IRATE:
                # Correct basis AND correct value
                self.eve_measurements_raw.append(self.alice_bits_raw[i])
                self.eve_had_correct_basis_raw.append(True)

            elif r < IRATE + ERATE/3:
                # Case 1: Wrong value, right basis
                self.eve_measurements_raw.append(1 - self.alice_bits_raw[i])
                self.eve_had_correct_basis_raw.append(True)

            elif r < IRATE + 2*ERATE/3:
                # Case 2: Wrong value, wrong basis
                self.eve_measurements_raw.append(1 - self.alice_bits_raw[i])
                self.eve_had_correct_basis_raw.append(False)

            else:
                # Case 3: Right value, wrong basis
                self.eve_measurements_raw.append(self.alice_bits_raw[i])
                self.eve_had_correct_basis_raw.append(False)

        # Bob receives what Eve sends (intercept-resend)
        self.bob_bits_raw = []
        for i in range(raw_key_size):
            if self.eve_had_correct_basis_raw[i]:
                # Eve had correct basis - Bob gets Eve's measurement
                self.bob_bits_raw.append(self.eve_measurements_raw[i])
            else:
                # Eve had wrong basis - Bob gets random bit (50/50)
                self.bob_bits_raw.append(random.randint(0, 1))

        # Perform basis sifting (keep only matching Alice-Bob bases)
        self.perform_sifting()

    def perform_sifting(self):
        """Perform basis sifting - keep only positions where Alice and Bob used same basis."""
        self.alice_bits = []
        self.bob_bits = []
        self.eve_measurements = []
        self.eve_had_correct_basis = []
        self.sifted_indices = []

        for i in range(self.raw_key_size):
            if self.alice_bases[i] == self.bob_bases[i]:
                self.sifted_indices.append(i)
                self.alice_bits.append(self.alice_bits_raw[i])
                self.bob_bits.append(self.bob_bits_raw[i])
                self.eve_measurements.append(self.eve_measurements_raw[i])
                self.eve_had_correct_basis.append(self.eve_had_correct_basis_raw[i])

        self.sifted_key_size = len(self.alice_bits)

        # Create Key objects for CASCADE
        self.alice_key = Key()
        self.alice_key._size = self.sifted_key_size
        self.alice_key._bits = self.alice_bits.copy()

        self.bob_key = Key()
        self.bob_key._size = self.sifted_key_size
        self.bob_key._bits = self.bob_bits.copy()

    def calculate_qber(self):
        """Calculate QBER between Alice and Bob after sifting."""
        errors = sum(1 for i in range(self.sifted_key_size)
                    if self.alice_bits[i] != self.bob_bits[i])
        qber = errors / self.sifted_key_size if self.sifted_key_size > 0 else 0
        return qber

    def verify_eve_bob_correlation(self):
        """Verify that Eve knows Bob's bits when basis was correct."""
        correlations = []

        for i in range(self.sifted_key_size):
            if self.eve_had_correct_basis[i]:
                # When Eve had correct basis, Eve's bit should equal Bob's bit
                if self.eve_measurements[i] == self.bob_bits[i]:
                    correlations.append(True)
                else:
                    correlations.append(False)
                    print(f"WARNING: Eve-Bob mismatch at position {i} despite correct basis!")

        if correlations:
            correlation_rate = sum(correlations) / len(correlations)
            print(f"Eve-Bob correlation when basis correct: {correlation_rate:.1%} (should be 100%)")

        return correlations

    def run_cascade(self):
        """Run CASCADE error correction and collect parity constraints."""
        constraints = []

        class ConstraintCollector(MockClassicalChannel):
            def __init__(self, alice_key, constraints_list):
                super().__init__(alice_key)
                self.constraints = constraints_list

            def ask_parities(self, blocks):
                alice_parities = super().ask_parities(blocks)

                for block, alice_parity in zip(blocks, alice_parities):
                    indices = list(block.get_key_indexes())
                    self.constraints.append({
                        'indices': indices,
                        'alice_parity': alice_parity,  # Eve observes Alice's parity sent to Bob
                        'size': len(indices)
                    })

                return alice_parities

        channel = ConstraintCollector(self.alice_key, constraints)
        # Use the expected QBER for CASCADE (2*ERATE/3)
        expected_qber = 2 * ERATE / 3
        reconciliation = Reconciliation("original", channel, self.bob_key, expected_qber)

        try:
            reconciliation.reconcile()
        except:
            pass  # May fail if error rate is too high

        return constraints


class EveQuadratureAttack:

    def __init__(self, model, constraints):
        """
        Initialize Eve's attack.

        Args:
            model: QuadratureAttackModel instance
            constraints: CASCADE parity constraints
        """
        self.model = model
        self.constraints = constraints
        self.key_size = model.sifted_key_size

        # Indices where Eve knows Bob exactly (correct basis)
        self.eve_knows_bob = [i for i in range(self.key_size)
                              if model.eve_had_correct_basis[i]]
        self.eve_uncertain = [i for i in range(self.key_size)
                             if not model.eve_had_correct_basis[i]]

        print(f"\nEve's knowledge:")
        print(f"  Knows Bob exactly: {len(self.eve_knows_bob)}/{self.key_size} ({len(self.eve_knows_bob)*100/self.key_size:.1f}%)")
        print(f"  Uncertain about Bob: {len(self.eve_uncertain)}/{self.key_size} ({len(self.eve_uncertain)*100/self.key_size:.1f}%)")

    def solve_constraints(self):
        """
        Solve for Alice's bits

        Key constraints:
        1. When Eve had correct basis: Eve_bit = Bob_bit, and ~3.3% differ from Alice
        2. When Eve had wrong basis: Bob's bit is random, 50% match with Alice
        3. CASCADE parity constraints
        4. Overall QBER = 2*ERATE/3
        """

        start_time = time.time()

        print(f"\n=== SOLVING CONSTRAINTS ===")
        print(f"Key size: {self.key_size} bits")
        print(f"CASCADE constraints: {len(self.constraints)}")
        print(f"Expected QBER: {2*ERATE/3:.4f} ({2*ERATE/3*100:.2f}%)")

        # Create SAT solver
        solver = Solver()
        solver.set('timeout', 30000)  # 30 second timeout

        # Variables for Alice's bits
        alice_vars = [Bool(f'alice_{i}') for i in range(self.key_size)]

        # Add CASCADE parity constraints
        print("Adding CASCADE constraints...")
        for c in self.constraints:
            indices = c['indices']
            alice_parity = c['alice_parity']

            if len(indices) == 1:
                solver.add(alice_vars[indices[0]] == (alice_parity == 1))
            else:
                xor_expr = alice_vars[indices[0]]
                for idx in indices[1:]:
                    xor_expr = Xor(xor_expr, alice_vars[idx])
                solver.add(xor_expr == (alice_parity == 1))

        # Constraint 1: When Eve had correct basis (IRATE + ERATE/3 of sifted bits)
        # Eve's bit = Bob's bit, and (ERATE/3)/(IRATE + ERATE/3) differ from Alice
        if self.eve_knows_bob:
            errors_correct_basis = Sum([
                If(alice_vars[i] != (self.model.bob_bits[i] == 1), 1, 0)
                for i in self.eve_knows_bob
            ])

            # Expected errors: ERATE/3 portion of correct basis bits
            expected_errors = int(len(self.eve_knows_bob) * (ERATE/3) / (IRATE + ERATE/3))
            tolerance = max(2, int(len(self.eve_knows_bob) * 0.02))

            solver.add(errors_correct_basis >= max(0, expected_errors - tolerance))
            solver.add(errors_correct_basis <= expected_errors + tolerance)

            print(f"  Correct basis constraint: {expected_errors}±{tolerance} errors among {len(self.eve_knows_bob)} bits")

        # Constraint 2: When Eve had wrong basis (2*ERATE/3 of sifted bits)
        # Bob's bit is random, 50% match with Alice
        if self.eve_uncertain:
            errors_wrong_basis = Sum([
                If(alice_vars[i] != (self.model.bob_bits[i] == 1), 1, 0)
                for i in self.eve_uncertain
            ])

            expected_errors = int(len(self.eve_uncertain) * 0.5)
            tolerance = max(3, int(len(self.eve_uncertain) * 0.15))

            solver.add(errors_wrong_basis >= max(0, expected_errors - tolerance))
            solver.add(errors_wrong_basis <= expected_errors + tolerance)

            print(f"  Wrong basis constraint: {expected_errors}±{tolerance} errors among {len(self.eve_uncertain)} bits")

        # Constraint 3: Overall QBER = 2*ERATE/3
        total_errors = Sum([
            If(alice_vars[i] != (self.model.bob_bits[i] == 1), 1, 0)
            for i in range(self.key_size)
        ])

        expected_qber_count = int(self.key_size * 2 * ERATE / 3)
        qber_tolerance = max(5, int(self.key_size * 0.01))

        solver.add(total_errors >= expected_qber_count - qber_tolerance)
        solver.add(total_errors <= expected_qber_count + qber_tolerance)

        print(f"  QBER constraint: {expected_qber_count}±{qber_tolerance} total errors")

        # Solve
        print("\nSolving SAT problem...")

        if solver.check() == sat:
            print("Solution found!")
            model = solver.model()

            # Extract solution
            alice_solution = []
            for i in range(self.key_size):
                val = model.eval(alice_vars[i])
                alice_solution.append(1 if is_true(val) else 0)

            # Calculate accuracy
            correct = sum(1 for i in range(self.key_size)
                         if alice_solution[i] == self.model.alice_bits[i])
            accuracy = correct / self.key_size

            # Baseline accuracy (using Eve's measurements)
            baseline_correct = sum(1 for i in range(self.key_size)
                                 if self.model.eve_measurements[i] == self.model.alice_bits[i])
            baseline_acc = baseline_correct / self.key_size

            solve_time = time.time() - start_time

            print(f"\n=== RESULTS ===")
            print(f"Baseline accuracy: {baseline_acc:.4f}")
            print(f"SAT solver accuracy: {accuracy:.4f}")
            print(f"Improvement: {accuracy - baseline_acc:.4f}")
            print(f"Bits recovered: {correct}/{self.key_size}")
            print(f"Remaining errors: {self.key_size - correct}")
            print(f"Solve time: {solve_time:.3f}s")

            if self.key_size - correct < 128:
                print("\n⚠️  KEY IS CRYPTOGRAPHICALLY BROKEN!")

            return accuracy, solve_time

        else:
            print("No solution found!")
            solve_time = time.time() - start_time
            return 0.0, solve_time


def test_idea(raw_key_size=2048, seed=42):
    print("="*70)
    print(f"IRATE = {IRATE:.1%}")
    print(f"ERATE = {ERATE:.1%}")
    print(f"Expected QBER = 2*ERATE/3 = {2*ERATE/3:.4f} ({2*ERATE/3*100:.2f}%)")
    print(f"Raw key size: {raw_key_size} bits")
    print()

    # Create model
    model = QuadratureAttackModel(raw_key_size, seed)

    print(f"After sifting: {model.sifted_key_size} bits (~{model.sifted_key_size*100/raw_key_size:.1f}%)")

    # Calculate and verify QBER
    qber = model.calculate_qber()
    expected_qber = 2 * ERATE / 3

    print(f"\nQBER verification:")
    print(f"  Measured QBER: {qber:.4f} ({qber*100:.2f}%)")
    print(f"  Expected QBER: {expected_qber:.4f} ({expected_qber*100:.2f}%)")

    if abs(qber - expected_qber) < 0.01:
        print("  ✓ QBER matches expected value!")
    else:
        print(f"  ⚠️  QBER deviation: {abs(qber - expected_qber):.4f}")

    # Verify Eve-Bob correlation
    print("\nVerifying Eve-Bob correlation...")
    model.verify_eve_bob_correlation()

    # Run CASCADE
    print("\nRunning CASCADE...")
    constraints = model.run_cascade()
    print(f"Collected {len(constraints)} parity constraints")
    print(f"Information leakage: {len(constraints)*100/model.sifted_key_size:.1f}% of sifted key size")

    # Run Eve's attack
    attacker = EveQuadratureAttack(model, constraints)
    accuracy, solve_time = attacker.solve_constraints()

    return accuracy, qber


if __name__ == "__main__":
    # Test with different key sizes
    for raw_size in [512, 1024, 2048, 4096]:
        print("\n" + "="*70)
        for seedrange in range(3):
            accuracy, qber = test_idea(raw_size, seed=42+seedrange)
            print(f"\nFinal: {raw_size} raw bits → accuracy={accuracy:.4f}, QBER={qber:.4f}")
