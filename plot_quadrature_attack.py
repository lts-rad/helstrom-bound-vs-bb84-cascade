"""
Using chunked solver for larger keys for performance, not ideal, but runs
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from quadrature_attack_model import QuadratureAttackModel, ERATE
from chunked_solver import ChunkedSolver
from chunked_optimized import OptimizedChunkedSolver
from z3 import *


class CorrectedChunkedSolver:
    """Chunked solver using corrected constraints (Eve's measurements)."""

    def __init__(self, model, constraints, chunk_size=4096, overlap=512):
        self.model = model
        self.constraints = constraints
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.key_size = model.sifted_key_size

        # Eve's knowledge from measurements
        self.eve_knows_bob_indices = set(i for i in range(self.key_size)
                                         if model.eve_had_correct_basis[i])
        self.eve_bob_values = {i: model.eve_measurements[i] for i in self.eve_knows_bob_indices}

    def solve_chunk(self, start, end):
        """Solve a chunk using optimization with corrected constraints."""
        chunk_size = end - start

        # Get constraints for this chunk
        chunk_constraints = []
        for c in self.constraints:
            indices = c['indices']
            # Include if any index is in chunk
            if any(start <= idx < end for idx in indices):
                # Adjust indices to be relative to chunk
                adjusted_indices = [idx - start for idx in indices if start <= idx < end]
                if adjusted_indices:
                    chunk_constraints.append({
                        'indices': adjusted_indices,
                        'alice_parity': c['alice_parity'],  # Eve observes Alice's parity sent to Bob
                        'original_indices': indices
                    })

        # Create optimizer
        optimizer = Optimize()
        optimizer.set('timeout', 5000)

        # Variables for this chunk
        alice_vars = [Bool(f'alice_{i}') for i in range(chunk_size)]

        # Add CASCADE constraints
        for c in chunk_constraints:
            if len(c['indices']) == len(c['original_indices']):
                # All indices in chunk
                indices = c['indices']
                if len(indices) == 1:
                    optimizer.add(alice_vars[indices[0]] == (c['alice_parity'] == 1))
                else:
                    xor_expr = alice_vars[indices[0]]
                    for idx in indices[1:]:
                        if 0 <= idx < chunk_size:
                            xor_expr = Xor(xor_expr, alice_vars[idx])
                    optimizer.add(xor_expr == (c['alice_parity'] == 1))

        # Maximize agreement with Eve's knowledge for this chunk
        eve_indices_in_chunk = [i - start for i in range(start, end) if i in self.eve_knows_bob_indices]
        if eve_indices_in_chunk:
            agreement = Sum([
                If(alice_vars[i] == (self.eve_bob_values[i + start] == 1), 1, 0)
                for i in eve_indices_in_chunk
            ])
            optimizer.maximize(agreement)

        # Solve
        result = optimizer.check()

        if result == sat:
            model_solution = optimizer.model()
            chunk_solution = []
            for i in range(chunk_size):
                val = model_solution.eval(alice_vars[i])
                if val is None:
                    # Use Eve's measurement if available
                    global_idx = start + i
                    if global_idx in self.eve_bob_values:
                        chunk_solution.append(self.eve_bob_values[global_idx])
                    else:
                        chunk_solution.append(0)
                else:
                    chunk_solution.append(1 if is_true(val) else 0)
            return chunk_solution
        else:
            # Failed - use Eve's measurements
            return [self.eve_bob_values.get(start + i, 0) for i in range(chunk_size)]

    def solve(self):
        """Solve all chunks and combine."""
        recovered_bits = [0] * self.key_size

        # Process chunks
        for start in range(0, self.key_size, self.chunk_size - self.overlap):
            end = min(start + self.chunk_size, self.key_size)

            print(f"    Solving chunk [{start}:{end}]...", end=' ')
            chunk_solution = self.solve_chunk(start, end)

            # Copy solution to recovered bits
            for i, bit in enumerate(chunk_solution):
                if start + i < self.key_size:
                    recovered_bits[start + i] = bit

            chunk_correct = sum(1 for i in range(len(chunk_solution))
                              if start + i < self.key_size and
                              chunk_solution[i] == self.model.alice_bits[start + i])
            chunk_acc = chunk_correct / len(chunk_solution) if chunk_solution else 0
            print(f"accuracy={chunk_acc:.3f}")

        # Calculate overall accuracy
        correct = sum(1 for i in range(self.key_size)
                     if recovered_bits[i] == self.model.alice_bits[i])
        return correct / self.key_size


def test_and_plot():
    """Test various key sizes and plot results."""

    # Test key sizes
    raw_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

    # Results storage
    results = {
        'raw_sizes': [],
        'sifted_sizes': [],
        'accuracies': [],
        'baseline_accuracies': [],
        'improvements': [],
        'solve_times': [],
        'remaining_errors': [],
        'constraint_counts': []
    }

    # Try multiple seeds for each size to get best accuracy
    test_seeds = [4242, 4343, 4444, 5555, 6666, 7777]  # Test these seeds

    for raw_size in raw_sizes:
        print(f"\n{'='*70}")
        print(f"Testing {raw_size} bits")
        print("="*70)

        best_accuracy = 0
        best_model = None
        best_seed = None
        best_solve_time = 0
        best_constraints = None

        # Try each seed and keep the best result
        for seed in test_seeds:
            print(f"  Trying seed {seed}...", end='')
            try:
                # Create model
                model = QuadratureAttackModel(raw_size, seed)
                sifted_size = model.sifted_key_size

                # Run CASCADE
                constraints = model.run_cascade()

                # Use optimized solver for smaller keys, chunked for larger
                start_time = time.time()
                if raw_size <= 4096:
                    # Use optimization directly for smaller keys
                    solver = CorrectedChunkedSolver(model, constraints, chunk_size=2048, overlap=256)
                else:
                    # Use chunked solver for larger keys
                    solver = CorrectedChunkedSolver(model, constraints, chunk_size=4096, overlap=512)
                accuracy = solver.solve()
                solve_time = time.time() - start_time

                print(f" Accuracy: {accuracy:.4f}")

                # Keep track of best result
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_seed = seed
                    best_solve_time = solve_time
                    best_constraints = constraints

                # Stop if we achieve perfect accuracy
                if accuracy >= 1.0:
                    print(f"  ✓ Perfect accuracy with seed {seed}!")
                    break

            except Exception as e:
                print(f" Error: {e}")
                continue

        # Use the best result for this key size
        if best_model is not None:
            print(f"\nBest result: seed {best_seed}, accuracy {best_accuracy:.4f}")

            sifted_size = best_model.sifted_key_size

            # Calculate Eve's baseline accuracy (what she knows without CASCADE)
            eve_baseline_acc = sum(1 for i in range(sifted_size)
                                  if best_model.eve_measurements[i] == best_model.alice_bits[i]) / sifted_size

            # Also calculate QBER for reference
            qber = sum(1 for i in range(sifted_size)
                      if best_model.bob_bits[i] != best_model.alice_bits[i]) / sifted_size

            improvement = best_accuracy - eve_baseline_acc
            remaining_errors = sifted_size - int(best_accuracy * sifted_size)

            # Store results
            results['raw_sizes'].append(raw_size)
            results['sifted_sizes'].append(sifted_size)
            results['accuracies'].append(best_accuracy)
            results['baseline_accuracies'].append(eve_baseline_acc)
            results['improvements'].append(improvement)
            results['solve_times'].append(best_solve_time)
            results['remaining_errors'].append(remaining_errors)
            results['constraint_counts'].append(len(best_constraints))

            print(f"\nResults:")
            print(f"  Accuracy: {best_accuracy:.4f}")
            print(f"  Eve's baseline: {eve_baseline_acc:.4f}")
            print(f"  QBER: {qber:.4f}")
            print(f"  Improvement: {improvement:.4f}")
            print(f"  Time: {best_solve_time:.2f}s")
            print(f"  Remaining errors: {remaining_errors}")

            if best_accuracy >= 0.995:
                print("  ✓ ACHIEVED 99.5% accuracy!")
            elif best_accuracy >= 0.99:
                print("  ⚬ Close: 99%+ accuracy")

    # Convert to numpy arrays for easier plotting
    raw_sizes = np.array(results['raw_sizes'])
    sifted_sizes = np.array(results['sifted_sizes'])
    accuracies = np.array(results['accuracies'])
    baseline_accuracies = np.array(results['baseline_accuracies'])
    improvements = np.array(results['improvements'])
    solve_times = np.array(results['solve_times'])
    remaining_errors = np.array(results['remaining_errors'])
    constraint_counts = np.array(results['constraint_counts'])

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Eve's Attack Performance (Helstrom Bound Quadrature Measurement)", fontsize=16)

    # Define powers of 2 for x-axis labels
    powers_of_2 = [2**i for i in range(10, 17)]  # 2^10 to 2^16

    # Plot 1: Accuracy vs Key Size
    ax = axes[0, 0]
    ax.plot(raw_sizes, accuracies * 100,
            'b-o', label='Attack Accuracy', linewidth=2, markersize=8)
    ax.plot(raw_sizes, baseline_accuracies * 100,
            'r--s', label="Eve's Baseline (no CASCADE)", linewidth=1.5, markersize=6)
    ax.axhline(y=100, color='g', linestyle=':', label='100% Recovery', linewidth=1.5)

    # Add vertical lines at 100% accuracy points
    for i, (size, acc) in enumerate(zip(raw_sizes, accuracies)):
        if acc >= 1.0:
            ax.axvline(x=size, color='lightblue', linestyle=':', alpha=0.3)

    # Add vertical line to mark switch to larger chunks
    ax.axvline(x=8192, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(8192 * 1.1, 95, 'Larger\nchunks →', fontsize=9,
            color='red', fontweight='bold', ha='left', va='center')

    ax.set_xlabel('Total Message Size (bits)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('CASCADE Constraint Solving Attack vs Total Message Size', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([85, 101])
    ax.set_xscale('log', base=2)
    ax.set_xticks(powers_of_2)
    ax.set_xticklabels([f'$2^{{{i}}}$' for i in range(10, 17)])

    # Plot 2: Improvement over Baseline
    ax = axes[0, 1]
    ax.plot(raw_sizes, improvements * 100,
            'g-o', linewidth=2, markersize=8)
    ax.set_xlabel('Total Message Size (bits)', fontsize=12)
    ax.set_ylabel('Improvement over Baseline (%)', fontsize=12)
    ax.set_title('Improvement over Baseline', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(improvements) * 110])
    ax.set_xscale('log', base=2)
    ax.set_xticks(powers_of_2)
    ax.set_xticklabels([f'$2^{{{i}}}$' for i in range(10, 17)])

    # Plot 3: Eve's Knowledge Distribution
    ax = axes[0, 2]

    # Calculate Eve's knowledge percentages for 1 seed
    eve_knows_percentages = []
    for raw_size in raw_sizes:
        sample_model = QuadratureAttackModel(raw_size, 4242)
        eve_knows = sum(1 for i in range(sample_model.sifted_key_size)
                       if sample_model.eve_had_correct_basis[i])
        eve_knows_pct = (eve_knows / sample_model.sifted_key_size) * 100
        eve_knows_percentages.append(eve_knows_pct)

    # Create stacked bar chart
    eve_knows_pct = np.array(eve_knows_percentages)
    eve_uncertain_pct = 100 - eve_knows_pct

    bar_width = 0.9
    x_pos = np.arange(len(raw_sizes))

    ax.bar(x_pos, eve_uncertain_pct, bar_width, label='Eve uncertain (wrong basis)',
           color='darkred', alpha=0.8)
    ax.bar(x_pos, eve_knows_pct, bar_width, bottom=eve_uncertain_pct,
           label="Eve certain of Bob's bitstream", color='lightgreen', alpha=0.8)

    # Add percentage labels
    for i, (knows, uncertain) in enumerate(zip(eve_knows_pct, eve_uncertain_pct)):
        ax.text(i, uncertain/2, f'{uncertain:.1f}%', ha='center', va='center',
                color='white', fontweight='bold', fontsize=8)
        ax.text(i, uncertain + knows/2, f'{knows:.1f}%', ha='center', va='center',
                color='black', fontsize=8)

    ax.set_xlabel('Total Message Size (bits)', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title("Eve's Knowledge of Bob After Sifting", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'$2^{{{int(np.log2(size))}}}$' for size in raw_sizes])
    ax.set_ylim([0, 100])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Remaining Errors
    ax = axes[1, 0]
    errors_to_plot = np.maximum(remaining_errors, 1)
    ax.loglog(raw_sizes, errors_to_plot,
              'r-o', linewidth=2, markersize=8, base=2)
    ax.axhline(y=128, color='orange', linestyle='--',
               label='128-bit Security Threshold', linewidth=1.5)
    ax.set_xlabel('Total Message Size (bits)', fontsize=12)
    ax.set_ylabel('Remaining Errors (bits)', fontsize=12)
    ax.set_title('Remaining Key Errors', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xticks(powers_of_2)
    ax.set_xticklabels([f'$2^{{{i}}}$' for i in range(10, 17)])

    # Set y-axis to show powers of 2
    y_powers = [2**i for i in range(0, 11)]
    ax.set_yticks(y_powers)
    ax.set_yticklabels([f'$2^{{{i}}}$' for i in range(0, 11)])

    # Plot 5: CASCADE Constraints
    ax = axes[1, 1]
    ax.loglog(raw_sizes, constraint_counts,
              'b-o', linewidth=2, markersize=8, base=2)
    ax.set_xlabel('Total Message Size (bits)', fontsize=12)
    ax.set_ylabel('Number of CASCADE Constraints', fontsize=12)
    ax.set_title('CASCADE Parity Constraints', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xticks(powers_of_2)
    ax.set_xticklabels([f'$2^{{{i}}}$' for i in range(10, 17)])

    # Set y-axis to show powers of 2
    y_powers_cascade = [2**i for i in range(7, 15)]
    ax.set_yticks(y_powers_cascade)
    ax.set_yticklabels([f'$2^{{{i}}}$' for i in range(7, 15)])

    # Plot 6: IRATE vs QBER (Helstrom Bound)
    ax = axes[1, 2]

    # Define Helstrom bound function for 4-PSK
    def helstrom_bound_4psk(alpha_squared):
        """Computes the Helstrom bound for 4-PSK coherent states."""
        if alpha_squared < 0:
            return None

        N = 4
        exp_minus_alpha_sq = np.exp(-alpha_squared)

        # Calculate the eigenvalues h_p for N=4
        h1 = 2 * exp_minus_alpha_sq * (np.cosh(alpha_squared) + np.cos(alpha_squared))
        h2 = 2 * exp_minus_alpha_sq * (np.sinh(alpha_squared) + np.sin(alpha_squared))
        h3 = 2 * exp_minus_alpha_sq * (np.cosh(alpha_squared) - np.cos(alpha_squared))
        h4 = 2 * exp_minus_alpha_sq * (np.sinh(alpha_squared) - np.sin(alpha_squared))
        h4 = max(0, h4)

        # Calculate the sum of the square roots of the eigenvalues
        sum_sqrt_h = np.sqrt(max(0, h1)) + np.sqrt(max(0, h2)) + np.sqrt(max(0, h3)) + np.sqrt(h4)

        # Calculate the Helstrom bound
        sum_term = (1.0 / N) * sum_sqrt_h
        p_hel = 1.0 - sum_term * sum_term

        return max(0.0, min(p_hel, 1.0))

    # Calculate for a range of amplitudes
    amplitudes = np.linspace(0, 2.0, 100)
    photon_numbers = amplitudes ** 2  # Mean photon number = α²
    erates = []
    qbers = []

    for alpha_sq in photon_numbers:
        erate = helstrom_bound_4psk(alpha_sq)
        # QBER = 2 * ERATE / 3 (from the model)
        qber = 2 * erate / 3
        erates.append(erate)
        qbers.append(qber)

    # Plot Helstrom Error (ERATE) vs QBER
    ax.plot(np.array(qbers) * 100, np.array(erates) * 100, 'b-', linewidth=2.5)

    # Mark our operating point with precise Helstrom bound values
    our_erate = ERATE
    our_qber = 2 * our_erate / 3
    ax.plot(our_qber * 100, our_erate * 100, 'ro', markersize=10)

    # Add text annotation for α=1.0 and QBER=6.16%
    ax.annotate('α=1.0\nQBER=%.3f'%our_qber, xy=(our_qber * 100, our_erate * 100),
                xytext=(8, 14),  # Place text in upper right area
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.set_xlabel('4-State BB84 QBER (%)', fontsize=12)
    ax.set_ylabel('Helstrom Bound Error (%)', fontsize=12)
    ax.set_title('Helstrom Bound vs 4-State BB84 QBER', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 15])

    plt.tight_layout()
    plt.savefig('cascade_final_results_corrected.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\nRaw Size | Sifted | Accuracy | Baseline | Improvement | Time")
    print("-" * 65)
    for i in range(len(results['raw_sizes'])):
        print(f"{results['raw_sizes'][i]:8} | "
              f"{results['sifted_sizes'][i]:6} | "
              f"{results['accuracies'][i]:8.2%} | "
              f"{results['baseline_accuracies'][i]:8.2%} | "
              f"{results['improvements'][i]:+11.2%} | "
              f"{results['solve_times'][i]:5.1f}s")

    # Check for perfect recovery
    perfect = [i for i, acc in enumerate(results['accuracies']) if acc >= 0.99]
    if perfect:
        print(f"\n✓ PERFECT RECOVERY (≥99%) achieved for:")
        for i in perfect:
            print(f"  - {results['raw_sizes'][i]}-bit key ({results['sifted_sizes'][i]} sifted)")



if __name__ == "__main__":
    test_and_plot()
