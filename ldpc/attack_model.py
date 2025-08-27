"""
LDPC Attack Model: Eve observes syndrome and disclosed bits to recover Alice/Bob keys

This model simulates Eve's attack on LDPC error correction in QKD:
1. Eve observes the syndrome bits Alice sends to Bob
2. Eve observes the disclosed bits from interactive error correction
3. Eve performs LDPC correction to recover alice's codeword

Based on the quadrature_attack_model.py approach but adapted for LDPC constraints.
"""

import numpy as np
import random
import time
from numpy import copy
import error_correction_lib as ec
from file_utils import codes_from_file

# Helstrom bound parameters (same as quadrature model)
ERATE = 0.092421  # Error rate from 4-PSK Helstrom bound
IRATE = 1 - ERATE

class LDPCAttackModel:
    """Models Eve's attack on LDPC error correction"""

    def __init__(self, qber=0.06, code_file='codes_1944.txt', n=1944, seed=42):
        """
        Initialize LDPC attack model

        Args:
            qber: Quantum bit error rate between Alice and Bob
            code_file: LDPC codes file to use
            n: Block length
            seed: Random seed for reproducibility
        """
        self.qber = qber
        self.n = n
        self.seed = seed

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)

        # Load LDPC codes
        self.codes = codes_from_file(code_file)
        R_range = [code[0] for code in self.codes]

        # Choose code parameters based on QBER
        self.R, self.s_n, self.p_n = ec.choose_sp(qber, 1.0, R_range, n)
        self.k_n = n - self.s_n - self.p_n
        self.m = int((1-self.R) * n)

        # Get code structure
        code_params = self.codes[(self.R, n)]
        self.s_y_joins = code_params['s_y_joins']  # Parity check matrix rows
        self.y_s_joins = code_params['y_s_joins']  # Parity check matrix cols
        self.punct_list = code_params['punct_list']

        print(f"LDPC Attack Model initialized:")
        print(f"  QBER: {qber:.3f}")
        print(f"  Code rate R: {self.R:.4f}")
        print(f"  Block length: {n}")
        print(f"  Information bits k: {self.k_n}")
        print(f"  Syndrome bits m: {self.m}")
        print(f"  Shortened bits s_n: {self.s_n}")
        print(f"  Punctured bits p_n: {self.p_n}")

    def generate_alice_key(self):
        """Generate Alice's key only - Bob's will come from Eve's intercept"""

        # Generate Alice's information bits
        self.alice_info_bits = ec.generate_key(self.k_n)

        print(f"  Generated Alice's {self.k_n}-bit key")

        return self.alice_info_bits

    def simulate_eve_intercept(self):
        """
        Simulate Eve's quadrature interception using proper ERATE model from quadrature_attack_model.py

        ERATE = 0.092421, IRATE = 1 - ERATE
        Case breakdown:
        - IRATE: Correct basis AND correct value
        - ERATE/3: Wrong value, right basis
        - ERATE/3: Wrong value, wrong basis
        - ERATE/3: Right value, wrong basis
        """
        print(f"\n=== EVE'S INTERCEPTION (ERATE MODEL) ===")

        self.eve_measurements = []           # Eve's actual measurement results
        self.eve_had_correct_basis = []      # Whether Eve had correct basis (True/False)

        # Simulate Eve's quadrature measurement according to ERATE model
        for i in range(self.k_n):
            r = random.random()

            if r < IRATE:
                # Case 1: Correct basis AND correct value (90.76%)
                self.eve_measurements.append(self.alice_info_bits[i])
                self.eve_had_correct_basis.append(True)

            elif r < IRATE + ERATE/3:
                # Case 2: Wrong value, right basis (3.07%)
                self.eve_measurements.append(1 - self.alice_info_bits[i])
                self.eve_had_correct_basis.append(True)

            elif r < IRATE + 2*ERATE/3:
                # Case 3: Wrong value, wrong basis (3.07%)
                self.eve_measurements.append(1 - self.alice_info_bits[i])
                self.eve_had_correct_basis.append(False)

            else:
                # Case 4: Right value, wrong basis (3.07%)
                self.eve_measurements.append(self.alice_info_bits[i])
                self.eve_had_correct_basis.append(False)

        # Create Bob's bits based on Eve's intercept-resend (this is the ONLY source of errors)
        self.bob_info_bits = []
        for i in range(self.k_n):
            if self.eve_had_correct_basis[i]:
                # Eve had correct basis - Bob gets Eve's measurement
                self.bob_info_bits.append(self.eve_measurements[i])
            else:
                # Eve had wrong basis - Bob gets random bit (50/50)
                self.bob_info_bits.append(random.randint(0, 1))

        # Track actual error positions (Alice vs Bob)
        self.error_positions = []
        for i in range(self.k_n):
            if self.alice_info_bits[i] != self.bob_info_bits[i]:
                self.error_positions.append(i)

        # Indices where Eve knows Bob exactly (correct basis cases)
        self.eve_knows_bob = [i for i in range(self.k_n) if self.eve_had_correct_basis[i]]
        self.eve_uncertain = [i for i in range(self.k_n) if not self.eve_had_correct_basis[i]]

        actual_qber = len(self.error_positions) / self.k_n
        print(f"  ERATE model applied: {ERATE:.6f}")
        print(f"  Resulting Alice-Bob QBER: {actual_qber:.3f} ({actual_qber:.1%})")
        print(f"  Eve knows Bob exactly: {len(self.eve_knows_bob)}/{self.k_n} ({len(self.eve_knows_bob)*100/self.k_n:.1f}%)")
        print(f"  Eve uncertain about Bob: {len(self.eve_uncertain)}/{self.k_n} ({len(self.eve_uncertain)*100/self.k_n:.1f}%)")

        return self.alice_info_bits, self.bob_info_bits

    def perform_ec_with_tracking(self, x, y, s_y_joins, y_s_joins, qber_est, s_n, p_n, discl_n=20, show=0):
        """
        Simple wrapper around perform_ec that estimates disclosed positions
        """
        # Run the original working perform_ec function
        add_info, com_iters, x_dec, ver_check = ec.perform_ec(
            x, y, s_y_joins, y_s_joins, qber_est, s_n, p_n,
            punct_list=self.punct_list, discl_n=discl_n, show=show)

        # Estimate disclosed positions - we know add_info total bits were disclosed
        # For simplicity, assume the first add_info error positions were disclosed
        disclosed_positions = self.error_positions[:min(len(self.error_positions), add_info)]
        disclosed_values = [int(x[pos]) for pos in disclosed_positions]

        if show > 0:
            print(f"  Estimated {len(disclosed_positions)} disclosed info bit positions")

        return add_info, com_iters, x_dec, ver_check, disclosed_positions, disclosed_values

    def simulate_ldpc_error_correction(self):
        """
        Simulate LDPC error correction process and capture Eve's observations
        """
        # Generate position mappings
        s_pos, p_pos, k_pos = ec.generate_sp(self.s_n, self.p_n, self.k_n,
                                              p_list=self.punct_list)

        # Extend keys with shortened and punctured bits exactly as Bob does
        x_ext = ec.extend_sp(self.alice_info_bits, s_pos, p_pos, k_pos)
        y_ext = ec.extend_sp(self.bob_info_bits, s_pos, p_pos, k_pos)

        # Calculate syndromes (what Eve observes)
        s_x = ec.encode_syndrome(x_ext, self.s_y_joins)  # Alice's syndrome sent to bob
        s_y = ec.encode_syndrome(y_ext, self.s_y_joins)  # Bob's syndrome (not sent, private)

        # Simulate error correction process with disclosure tracking
        discl_n = int(round(self.n * (0.0280 - 0.02 * self.R)))

        print(f"\n=== LDPC Error Correction Process ===")
        print(f"  Syndrome bits transmitted: {len(s_x)}")
        print(f"  Disclosed bits per round: {discl_n}")

        # Run error correction with disclosure tracking
        add_info, com_iters, x_dec, ver_check, disclosed_positions, disclosed_values = self.perform_ec_with_tracking(
            self.alice_info_bits, self.bob_info_bits,
            self.s_y_joins, self.y_s_joins, self.qber,
            self.s_n, self.p_n, discl_n=discl_n, show=1)

        print(f"  Communication rounds: {com_iters}")
        print(f"  Total disclosed bits: {add_info}")
        print(f"  Disclosed info bit positions: {len(disclosed_positions)}")
        print(f"  Error correction successful: {ver_check}")

        # Debug: How many disclosed bits are actual error positions?
        disclosed_errors = sum(1 for pos in disclosed_positions if pos in self.error_positions)
        print(f"  DEBUG: Disclosed bits that are errors: {disclosed_errors}/{len(disclosed_positions)} ({disclosed_errors/len(disclosed_positions):.1%})")
        print(f"  DEBUG: Total error positions: {len(self.error_positions)}")
        print(f"  DEBUG: Errors found by disclosure: {disclosed_errors}/{len(self.error_positions)} ({disclosed_errors/len(self.error_positions):.1%})")

        # Sanity check: Verify Bob successfully recovered Alice's codeword
        if ver_check and x_dec is not None:
            # Compare Alice's original info bits with Bob's decoded result
            recovery_match = np.array_equal(self.alice_info_bits, x_dec)
            print(f"  SANITY CHECK: Bob recovered Alice's codeword: {recovery_match}")
            if not recovery_match:
                mismatches = np.sum(np.array(self.alice_info_bits) != np.array(x_dec))
                print(f"  WARNING: {mismatches}/{self.k_n} bits still differ after error correction")
        else:
            print(f"  SANITY CHECK: Error correction failed or no decoded result")

        # Store Eve's observations including disclosed bits and extended vector templates
        # Clear information bits from extended vectors so Eve doesn't accidentally use real values
        x_ext_template = x_ext.copy()
        y_ext_template = y_ext.copy()
        x_ext_template[k_pos] = 0  # Clear Alice's real info bits
        y_ext_template[k_pos] = 0  # Clear Bob's real info bits

        self.eve_observations = {
            'syndrome_alice': s_x,
            'syndrome_size': len(s_x),
            'disclosed_bits_per_round': discl_n,
            'communication_rounds': com_iters,
            'total_disclosed': add_info,
            'correction_successful': ver_check,
            'disclosed_positions': disclosed_positions,  # Information bit positions disclosed
            'disclosed_values': disclosed_values,        # Alice's actual bit values disclosed
            's_pos': s_pos,  # Exact shortened positions Bob used
            'p_pos': p_pos,  # Exact punctured positions Bob used
            'k_pos': k_pos,  # Exact information positions Bob used
            'alice_extended': x_ext_template,  # Extended vector template (info bits cleared)
            'bob_extended': y_ext_template     # Extended vector template (info bits cleared)
        }

        return self.eve_observations

    def recover_bob_key_ldpc(self):
        """
        Recover Bob's key by replicating Bob's exact LDPC process
        """
        print(f"\n=== EVE'S LDPC KEY RECOVERY ===")
        print(f"Key size: {self.k_n} bits")

        start_time = time.time()

        # Step 1: Realistic estimates - Eve uses her unambiguous measurements only
        # Eve constructs Bob estimate from her measurements
        eve_bob_estimate = []
        for i in range(self.k_n):
            if i in self.eve_knows_bob:
                eve_bob_estimate.append(self.eve_measurements[i])
            else:
                eve_bob_estimate.append(self.eve_measurements[i])
        eve_bob_estimate = np.array(eve_bob_estimate, dtype=int)

        # Update Bob estimate with disclosed positions
        # Use the disclosed values directly for Bob's estimate at those positions
        disclosed_positions = self.eve_observations['disclosed_positions']
        disclosed_values = self.eve_observations['disclosed_values']

        for pos, disclosed_value in zip(disclosed_positions, disclosed_values):
            if pos < self.k_n:
                # Set Bob's estimate to the disclosed value directly
                eve_bob_estimate[pos] = disclosed_value

        # Eve constructs Alice estimate from disclosed bits + her unambiguous measurements
        eve_alice_estimate = np.zeros(self.k_n, dtype=int)
        disclosed_positions = self.eve_observations['disclosed_positions']
        disclosed_values = self.eve_observations['disclosed_values']

        # Start with disclosed Alice bits
        for pos, value in zip(disclosed_positions, disclosed_values):
            if pos < self.k_n:
                eve_alice_estimate[pos] = value

        # For positions Eve knows unambiguously, use Alice = Bob (no error)
        # For positions Eve is uncertain, use her best guess
        for i in range(self.k_n):
            if i not in disclosed_positions:  # Not disclosed
                if i in self.eve_knows_bob:
                    # Eve knows Bob exactly, assume Alice = Bob (no error at this position)
                    eve_alice_estimate[i] = self.eve_measurements[i]
                else:
                    # Eve uncertain - use her measurement as best guess for Alice
                    eve_alice_estimate[i] = self.eve_measurements[i]

        print(f"  Using Eve's realistic Alice and Bob estimates")

        alice_estimate_accuracy = np.sum(eve_alice_estimate == self.alice_info_bits) / self.k_n
        bob_estimate_accuracy = np.sum(eve_bob_estimate == self.bob_info_bits) / self.k_n
        print(f"  Eve's Alice estimate accuracy: {alice_estimate_accuracy:.1%}")
        print(f"  Eve's Bob estimate accuracy: {bob_estimate_accuracy:.1%}")

        # Step 2: Use Eve's estimates for both Alice and Bob
        s_pos = self.eve_observations['s_pos']
        p_pos = self.eve_observations['p_pos']
        k_pos = self.eve_observations['k_pos']

        # Create Eve's Bob extended vector from her estimate
        bob_ext = self.eve_observations['bob_extended'].copy()
        bob_ext[k_pos] = eve_bob_estimate  # Replace with Eve's Bob estimate

        # Create Eve's Alice extended vector from her estimate
        alice_ext = self.eve_observations['alice_extended'].copy()
        alice_ext[k_pos] = eve_alice_estimate  # Replace with Eve's Alice estimate

        # Debug: Check if Eve's estimates correctly include disclosed positions
        disclosed_positions = self.eve_observations['disclosed_positions']
        disclosed_values = self.eve_observations['disclosed_values']

        alice_disclosed_correct = 0
        bob_disclosed_correct = 0
        for pos, true_alice_value in zip(disclosed_positions, disclosed_values):
            if pos < self.k_n:
                # Check Alice estimate at disclosed position
                if eve_alice_estimate[pos] == true_alice_value:
                    alice_disclosed_correct += 1

                # Check Bob estimate at disclosed position (should match real Bob)
                if eve_bob_estimate[pos] == true_alice_value:
                    bob_disclosed_correct += 1

        print(f"  DEBUG: Alice estimate includes {alice_disclosed_correct}/{len(disclosed_positions)} disclosed positions correctly")
        print(f"  DEBUG: Bob estimate includes {bob_disclosed_correct}/{len(disclosed_positions)} disclosed positions correctly")
        print(f"  Using Eve's Alice and Bob estimates")

        k_pos_in = copy(k_pos)  # For final exclusion (exactly like Bob)

        # Step 3: Use intercepted Alice syndrome and calculate Bob syndrome
        s_alice = self.eve_observations['syndrome_alice']  # Use intercepted syndrome from Alice
        s_bob = ec.encode_syndrome(bob_ext, self.s_y_joins)   # Calculate from Bob estimate
        print(f"  Using intercepted Alice syndrome ({len(s_alice)} bits)")

        s_d = (s_alice + s_bob) % 2  # Exactly like Bob: s_d = (s_x+s_y) % 2

        e_pat_in = ec.generate_key_zeros(self.n)  # Exactly like Bob

        # Step 4: Run LDPC without key_sum (realistic - Eve doesn't know Alice's full vector)
        discl_n = int(round(self.n * (0.0280 - 0.02 * self.R)))

        print(f"Running LDPC without key_sum (Eve doesn't know Alice's full vector)...")
        e_pat, minLLR_inds = ec.decode_syndrome_minLLR(
            e_pat_in, s_d, self.s_y_joins, self.y_s_joins,
            self.qber, s_pos, p_pos, k_pos,
            max_iter=100500, x=None, show=1, discl_n=discl_n, n_iter_avg_window=5
        )

        add_info = 0
        com_iters = 0

        # Step 5: Iterative process exactly like Bob
        while e_pat is None and com_iters < 10:
            print(f'  Additional iteration with p_n={len(p_pos)}, s_n={len(s_pos)}, k_n={len(k_pos)}')

            # Exactly like Bob: e_pat_in[minLLR_inds] = (x_ext[minLLR_inds]+y_ext[minLLR_inds]) % 2
            e_pat_in[minLLR_inds] = (alice_ext[minLLR_inds] + bob_ext[minLLR_inds]) % 2

            # Update position lists exactly like Bob
            s_pos = list(set(s_pos) | set(minLLR_inds))
            k_pos = list(set(k_pos) - set(minLLR_inds))
            if p_pos is not None:
                p_pos = list(set(p_pos) - set(minLLR_inds))

            # Decode again exactly like Bob
            e_pat, minLLR_inds = ec.decode_syndrome_minLLR(
                e_pat_in, s_d, self.s_y_joins, self.y_s_joins,
                self.qber, s_pos, p_pos, k_pos,
                r_start=None, max_iter=100500, x=None, show=1, discl_n=discl_n, n_iter_avg_window=5
            )

            add_info += discl_n
            com_iters += 1

        print(f"  LDPC: {com_iters} iterations, {add_info} additional disclosures")

        # Step 6: Apply result exactly like Bob
        if e_pat is not None:
            # LDPC found the error pattern to correct Bob estimate → Alice
            # The corrected result IS Alice (since that was the target syndrome)
            bob_ext_corrected = (bob_ext + e_pat) % 2

            # Extract Alice's information bits from the corrected result
            alice_recovered = bob_ext_corrected[k_pos_in]

            # Verify that we actually recovered Alice's true bits
            ver_check = np.array_equal(alice_recovered, self.alice_info_bits)
            alice_recovery_accuracy = np.sum(alice_recovered == self.alice_info_bits) / self.k_n
            print(f"  LDPC verification (recovered Alice == true Alice): {ver_check}")
            print(f"  Alice recovery accuracy: {alice_recovery_accuracy:.1%}")

            # For final scoring, we compare against Alice (since that's what we recovered)
            bob_solution = alice_recovered
            print(f"  Recovered Alice from LDPC-corrected result")
        else:
            print(f"  LDPC failed - no convergence")
            bob_solution = eve_bob_estimate.copy()
            ver_check = False

        solve_time = time.time() - start_time

        # Check final accuracy - we're recovering Alice, not Bob
        alice_correct = np.sum(bob_solution == self.alice_info_bits)

        print(f"\n=== EVE'S RECOVERY RESULTS ===")
        print(f"Alice's bits recovered: {alice_correct}/{self.k_n} ({alice_correct/self.k_n:.1%})")
        print(f"Processing time: {solve_time:.4f} seconds")

        return {
            'success': ver_check,
            'alice_accuracy': alice_correct / self.k_n,
            'solve_time': solve_time,
            'alice_solution': bob_solution,
            'ldpc_success': e_pat is not None,
        }

def main():
    """Run the LDPC attack simulation"""

    print("=== LDPC ATTACK SIMULATION ===")

    # Initialize attack model
    #attack = LDPCAttackModel(qber=0.06, code_file='codes_1944.txt', n=1944, seed=42)
    attack = LDPCAttackModel(qber=0.06, code_file='codes_4000.txt', n=4000, seed=42)

    # Generate Alice's key
    alice_bits = attack.generate_alice_key()

    # Simulate Eve's interception - this creates Bob's key with errors from Eve's process
    alice_bits, bob_bits = attack.simulate_eve_intercept()

    # Simulate LDPC error correction (Eve observes)
    observations = attack.simulate_ldpc_error_correction()

    # Recover Bob's key using LDPC decoding approach
    results = attack.recover_bob_key_ldpc()

    print(f"\n=== SUMMARY ===")
    if results['ldpc_success']:
        print(f"LDPC-based attack succeeded!")
        print(f"Eve recovered Alice's key with {results['alice_accuracy']:.1%} accuracy")
    else:
        print(f"LDPC decoding failed, using statistical fallback")
        print(f"Eve recovered Alice's key with {results['alice_accuracy']:.1%} accuracy")

    return results

if __name__ == "__main__":
    main()
