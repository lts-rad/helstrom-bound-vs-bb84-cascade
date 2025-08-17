"""
Chunked solver with heuristic methods for large keys.

Strategy:
1. Divide key into overlapping chunks (e.g., 1024-bit chunks with 128-bit overlap)
2. Solve each chunk independently
3. Merge solutions using overlap regions
4. Use heuristic refinement for inconsistencies
"""

import numpy as np
import time
from z3 import *
from quadrature_attack_model import QuadratureAttackModel, IRATE, ERATE

# Enable parallel solving globally for Z3
set_param('parallel.enable', True)
set_param('parallel.threads.max', 8)  # Use up to 8 threads


class ChunkedSolver:
    """Solve large keys by chunking."""

    def __init__(self, model, constraints, chunk_size=1024, overlap=128):
        self.model = model
        self.constraints = constraints
        self.key_size = model.sifted_key_size
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Where Eve knows Bob
        self.eve_knows_bob = set(i for i in range(self.key_size)
                                 if model.eve_had_correct_basis[i])

        print(f"\nChunked Solver Configuration:")
        print(f"  Key size: {self.key_size} bits")
        print(f"  Chunk size: {chunk_size} bits")
        print(f"  Overlap: {overlap} bits")
        print(f"  Eve knows Bob: {len(self.eve_knows_bob)} positions")

    def create_chunks(self):
        """Create overlapping chunks of the key."""
        chunks = []
        stride = self.chunk_size - self.overlap

        for start in range(0, self.key_size, stride):
            end = min(start + self.chunk_size, self.key_size)
            chunks.append((start, end))

            if end >= self.key_size:
                break

        print(f"  Created {len(chunks)} chunks")
        return chunks

    def get_chunk_constraints(self, start, end):
        """Get constraints that affect a chunk."""
        chunk_constraints = []

        for c in self.constraints:
            indices = c['indices']
            # Include constraint if any index is in chunk
            if any(start <= idx < end for idx in indices):
                # Filter to only indices in this chunk
                chunk_indices = [idx for idx in indices if start <= idx < end]
                if chunk_indices:
                    # Adjust indices to be relative to chunk
                    adjusted_indices = [idx - start for idx in chunk_indices]

                    # Only include if we have all indices (otherwise parity is incomplete)
                    if len(chunk_indices) == len(indices):
                        chunk_constraints.append({
                            'indices': adjusted_indices,
                            'alice_parity': c['alice_parity'],
                            'original_indices': chunk_indices
                        })

        return chunk_constraints

    def solve_chunk(self, start, end, initial_solution=None):
        """Solve a single chunk."""
        chunk_size = end - start
        chunk_constraints = self.get_chunk_constraints(start, end)

        print(f"\n  Chunk [{start}:{end}]: {chunk_size} bits, {len(chunk_constraints)} constraints")

        solver = Optimize()
        solver.set('timeout', 30000)  # 30s per chunk

        # Variables for this chunk
        alice_vars = [Bool(f'a_{i}') for i in range(chunk_size)]

        # Add CASCADE constraints for this chunk
        for c in chunk_constraints:
            indices = c['indices']
            alice_parity = c['alice_parity']

            if len(indices) == 1:
                solver.add(alice_vars[indices[0]] == (alice_parity == 1))
            else:
                xor_expr = alice_vars[indices[0]]
                for idx in indices[1:]:
                    xor_expr = Xor(xor_expr, alice_vars[idx])
                solver.add(xor_expr == (alice_parity == 1))

        # Add soft constraints based on Eve's knowledge
        for i in range(chunk_size):
            global_idx = start + i
            if global_idx in self.eve_knows_bob:
                bob_bit = self.model.bob_bits[global_idx]
                solver.add_soft(alice_vars[i] == (bob_bit == 1), weight=10)

        # If we have an initial solution (from overlap), use it
        if initial_solution is not None:
            for i, val in enumerate(initial_solution):
                if val is not None:
                    solver.add_soft(alice_vars[i] == (val == 1), weight=5)

        # Solve
        if solver.check() == sat:
            model_solution = solver.model()

            chunk_solution = []
            for i in range(chunk_size):
                val = model_solution.eval(alice_vars[i])
                chunk_solution.append(1 if is_true(val) else 0)

            return chunk_solution
        else:
            print(f"    Failed to solve chunk [{start}:{end}]")
            # Return Bob's bits as fallback
            return [self.model.bob_bits[start + i] for i in range(chunk_size)]

    def merge_chunks(self, chunk_solutions, chunks):
        """Merge chunk solutions, resolving conflicts in overlaps."""
        full_solution = [None] * self.key_size
        confidence = [0] * self.key_size

        for (start, end), solution in zip(chunks, chunk_solutions):
            for i, bit in enumerate(solution):
                global_idx = start + i

                if full_solution[global_idx] is None:
                    # First time seeing this bit
                    full_solution[global_idx] = bit
                    confidence[global_idx] = 1
                else:
                    # Overlap region - resolve conflict
                    if full_solution[global_idx] == bit:
                        # Agreement - increase confidence
                        confidence[global_idx] += 1
                    else:
                        # Disagreement - use higher confidence or keep existing
                        if confidence[global_idx] < 1:
                            full_solution[global_idx] = bit

        # Fill any remaining None values with Bob's bits
        for i in range(self.key_size):
            if full_solution[i] is None:
                full_solution[i] = self.model.bob_bits[i]

        return full_solution

    def heuristic_refinement(self, solution, max_iterations=10):
        """Refine solution using heuristic local search."""
        print("\n  Heuristic refinement...")

        best_solution = solution.copy()
        best_violations = self.count_violations(solution)

        for iteration in range(max_iterations):
            if best_violations == 0:
                break

            # Find violated constraints
            violated = []
            for c in self.constraints:
                indices = c['indices']
                alice_parity = c['alice_parity']
                computed_parity = sum(solution[i] for i in indices) % 2

                if computed_parity != alice_parity:
                    violated.append(c)

            if not violated:
                break

            # Pick a random violated constraint
            c = violated[np.random.randint(len(violated))]
            indices = c['indices']

            # Flip a random bit in this constraint
            # Prefer flipping bits where Eve is uncertain
            uncertain_in_block = [i for i in indices if i not in self.eve_knows_bob]

            if uncertain_in_block:
                flip_idx = uncertain_in_block[np.random.randint(len(uncertain_in_block))]
            else:
                flip_idx = indices[np.random.randint(len(indices))]

            # Flip the bit
            test_solution = solution.copy()
            test_solution[flip_idx] = 1 - test_solution[flip_idx]

            # Check if this improves things
            new_violations = self.count_violations(test_solution)

            if new_violations < best_violations:
                best_solution = test_solution
                best_violations = new_violations
                solution = test_solution
                print(f"    Iteration {iteration}: {best_violations} violations")

        return best_solution

    def count_violations(self, solution):
        """Count CASCADE constraint violations."""
        violations = 0
        for c in self.constraints:
            indices = c['indices']
            alice_parity = c['alice_parity']
            computed_parity = sum(solution[i] for i in indices) % 2

            if computed_parity != alice_parity:
                violations += 1

        return violations

    def solve(self):
        """Main solving routine using chunking."""
        start_time = time.time()

        print("\n=== CHUNKED SOLVER ===")

        # Create chunks
        chunks = self.create_chunks()

        # Solve each chunk
        chunk_solutions = []
        for i, (start, end) in enumerate(chunks):
            # For overlapping regions, use previous solution as hint
            initial = None
            if i > 0 and self.overlap > 0:
                # Get overlap from previous chunk
                prev_end = chunks[i-1][1]
                overlap_start = max(0, prev_end - self.overlap)
                if start < prev_end:
                    # Extract overlap solution
                    initial = [None] * (end - start)
                    for j in range(start, min(end, prev_end)):
                        if j - start < len(initial):
                            initial[j - start] = chunk_solutions[-1][j - chunks[i-1][0]]

            chunk_solution = self.solve_chunk(start, end, initial)
            chunk_solutions.append(chunk_solution)

        # Merge chunks
        print("\n  Merging chunks...")
        full_solution = self.merge_chunks(chunk_solutions, chunks)

        # Count initial violations
        initial_violations = self.count_violations(full_solution)
        print(f"  Initial violations: {initial_violations}/{len(self.constraints)}")

        # Heuristic refinement
        if initial_violations > 0:
            full_solution = self.heuristic_refinement(full_solution)
            final_violations = self.count_violations(full_solution)
            print(f"  Final violations: {final_violations}/{len(self.constraints)}")

        # Calculate accuracy
        if self.model.alice_bits:
            correct = sum(1 for i in range(self.key_size)
                        if full_solution[i] == self.model.alice_bits[i])
            accuracy = correct / self.key_size

            baseline_correct = sum(1 for i in range(self.key_size)
                                 if self.model.bob_bits[i] == self.model.alice_bits[i])
            baseline_acc = baseline_correct / self.key_size

            solve_time = time.time() - start_time

            print(f"\n=== RESULTS ===")
            print(f"Baseline (Bob=Alice): {baseline_acc:.4f}")
            print(f"Chunked solver accuracy: {accuracy:.4f}")
            print(f"Improvement: {accuracy - baseline_acc:.4f}")
            print(f"Solve time: {solve_time:.2f}s")

            remaining_errors = self.key_size - correct
            print(f"Remaining errors: {remaining_errors}")

            if remaining_errors < 128:
                print("⚠️  KEY IS CRYPTOGRAPHICALLY BROKEN!")

            return accuracy

        return 0.0


def test_chunked(raw_size=8192, seed=42):
    """Test chunked solver."""

    print("="*70)
    print("CHUNKED SOLVER TEST")
    print("="*70)

    # Create model
    model = QuadratureAttackModel(raw_size, seed)

    print(f"Raw key: {raw_size} bits")
    print(f"Sifted key: {model.sifted_key_size} bits")
    print(f"QBER: {model.calculate_qber():.4f}")

    # Run CASCADE
    constraints = model.run_cascade()
    print(f"CASCADE constraints: {len(constraints)}")

    # Test different chunk sizes
    for chunk_size in [512, 1024]:
        print(f"\n--- Chunk size: {chunk_size} ---")
        solver = ChunkedSolver(model, constraints, chunk_size=chunk_size, overlap=128)
        accuracy = solver.solve()

        if accuracy > 0.95:
            print(f"✓ SUCCESS with chunk size {chunk_size}")
            break

    return accuracy


if __name__ == "__main__":
    # Test on increasingly large sizes
    for raw_size in [16384, 32768, 65536]:
        print(f"\n{'='*70}")
        print(f"Testing {raw_size} bits")
        for seedrange in range(3):
            acc = test_chunked(raw_size, seed=42+seedrange)

            if acc < 0.95:
                print(f"Chunked solver not fully successful at {raw_size} bits")
                # Continue to see how far we can go
                # break
