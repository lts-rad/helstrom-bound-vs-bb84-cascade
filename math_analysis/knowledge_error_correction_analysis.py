import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
import random
import math

@dataclass
class BB84Simulation:
    """Simulates BB84 QKD protocol with Eve's Helstrom bound attack"""
    
    # Helstrom bound parameters from the document
    HELSTROM_ERROR_RATE = 0.092  # 9.2% minimum error for quadrature measurements
    CORRECT_MEASUREMENT = 0.908  # 90.8% correct measurements
    WRONG_BASIS_FRACTION = 2/3   # Of errors, 2/3 are wrong basis (6.12% of total)
    WRONG_VALUE_FRACTION = 1/3   # Of errors, 1/3 are wrong value (3.06% of total)
    
    def __init__(self, num_qubits: int = 10000, measurement_error: float = None, eve_sampling_rate: float = 1.0):
        self.num_qubits = num_qubits
        self.measurement_error = measurement_error if measurement_error is not None else self.HELSTROM_ERROR_RATE
        self.eve_sampling_rate = eve_sampling_rate
        self.alice_bits = None
        self.alice_bases = None
        self.bob_bases = None
        self.bob_bits = None
        self.eve_measurements = None
        self.sifted_key_indices = None
        
    def generate_random_bits(self, n: int) -> np.ndarray:
        """Generate n random bits (0 or 1)"""
        return np.random.randint(0, 2, n)
    
    def simulate_eve_quadrature_measurement(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate Eve's optimal quadrature measurement at Helstrom bound.
        Returns: (measurement_correct, error_type, sampled)
        error_type: 0 = no error, 1 = wrong basis, 2 = wrong value
        sampled: boolean array indicating which qubits Eve measured
        """
        # Determine which qubits Eve samples
        sampled = np.random.random(self.num_qubits) < self.eve_sampling_rate
        
        # Initialize arrays
        measurement_correct = np.ones(self.num_qubits, dtype=bool)  # Assume correct for unsampled
        error_type = np.zeros(self.num_qubits, dtype=int)
        
        # Only apply measurement errors to sampled qubits
        sampled_indices = np.where(sampled)[0]
        if len(sampled_indices) > 0:
            correct_rate = 1.0 - self.measurement_error
            measurement_correct[sampled_indices] = np.random.random(len(sampled_indices)) < correct_rate
            
            # For incorrect measurements on sampled qubits, determine error type
            incorrect_sampled = sampled_indices[~measurement_correct[sampled_indices]]
            for idx in incorrect_sampled:
                if np.random.random() < self.WRONG_BASIS_FRACTION:
                    error_type[idx] = 1  # Wrong basis error
                else:
                    error_type[idx] = 2  # Wrong value error
                
        return measurement_correct, error_type, sampled
    
    def run_protocol(self) -> dict:
        """Run the complete BB84 protocol with Eve's attack"""
        
        # Step 1: Alice prepares random bits in random bases
        self.alice_bits = self.generate_random_bits(self.num_qubits)
        self.alice_bases = self.generate_random_bits(self.num_qubits)
        
        # Step 2: Eve intercepts and measures using quadrature measurements
        eve_correct, eve_error_type, eve_sampled = self.simulate_eve_quadrature_measurement()
        
        # Step 3: Bob chooses random measurement bases
        self.bob_bases = self.generate_random_bits(self.num_qubits)
        
        # Step 4: Calculate Bob's measurement results based on Eve's interference
        self.bob_bits = np.zeros(self.num_qubits, dtype=int)
        
        for i in range(self.num_qubits):
            if not eve_sampled[i]:
                # Eve didn't sample this qubit - passes through undisturbed
                if self.alice_bases[i] == self.bob_bases[i]:
                    self.bob_bits[i] = self.alice_bits[i]
                else:
                    # Bases don't match - 50/50 chance
                    self.bob_bits[i] = np.random.randint(0, 2)
            elif eve_correct[i]:
                # Eve measured correctly - Bob gets correct bit if bases match
                if self.alice_bases[i] == self.bob_bases[i]:
                    self.bob_bits[i] = self.alice_bits[i]
                else:
                    # Bases don't match - 50/50 chance
                    self.bob_bits[i] = np.random.randint(0, 2)
            else:
                # Eve made an error
                if eve_error_type[i] == 1:  # Wrong basis error
                    # Bob gets 50/50 chance regardless
                    self.bob_bits[i] = np.random.randint(0, 2)
                else:  # Wrong value error
                    # Bob gets the wrong bit value
                    self.bob_bits[i] = 1 - self.alice_bits[i]
        
        # Step 5: Public basis announcement and sifting
        matching_bases = self.alice_bases == self.bob_bases
        self.sifted_key_indices = np.where(matching_bases)[0]
        
        # Calculate QBER on sifted key
        if len(self.sifted_key_indices) > 0:
            alice_sifted = self.alice_bits[self.sifted_key_indices]
            bob_sifted = self.bob_bits[self.sifted_key_indices]
            errors = alice_sifted != bob_sifted
            qber = np.mean(errors)
        else:
            qber = 0
            
        # Calculate Eve's knowledge after basis announcement
        eve_knowledge = self.calculate_eve_knowledge(eve_correct, eve_error_type, eve_sampled)
        
        # Calculate Alice's knowledge - she loses knowledge from all errors Eve introduces
        # Alice loses knowledge = Sample Rate × Measurement Error
        alice_knowledge = 1 - (self.eve_sampling_rate * self.measurement_error)
        
        return {
            'qber': qber,
            'eve_knowledge_percentage': eve_knowledge,
            'alice_knowledge_percentage': alice_knowledge,
            'sifted_key_length': len(self.sifted_key_indices),
            'total_qubits': self.num_qubits,
            'eve_correct_measurements': np.mean(eve_correct),
            'wrong_basis_errors': np.mean(eve_error_type == 1),
            'wrong_value_errors': np.mean(eve_error_type == 2)
        }
    
    def calculate_eve_knowledge(self, eve_correct: np.ndarray, 
                               eve_error_type: np.ndarray, eve_sampled: np.ndarray) -> float:
        """
        Calculate Eve's knowledge of Bob's bits after basis announcement.
        Eve only knows about qubits she sampled. For sampled qubits:
        - Correct measurements: she knows exactly
        - Wrong value errors: she can flip to get correct value
        - Wrong basis errors: no deterministic knowledge
        For unsampled qubits: no knowledge
        """
        # Eve's knowledge = sampling_rate * (1 - 2/3 * measurement_error)
        # She only knows about sampled qubits, and loses knowledge only from wrong basis errors
        eve_knowledge = self.eve_sampling_rate * (1 - (2/3 * self.measurement_error))
        return eve_knowledge
    
    def run_multiple_simulations(self, num_runs: int = 100) -> dict:
        """Run multiple simulations and return statistics"""
        results = {
            'qber_values': [],
            'eve_knowledge_values': []
        }
        
        for _ in range(num_runs):
            result = self.run_protocol()
            results['qber_values'].append(result['qber'])
            results['eve_knowledge_values'].append(result['eve_knowledge_percentage'])
            
        return {
            'mean_qber': np.mean(results['qber_values']),
            'std_qber': np.std(results['qber_values']),
            'mean_eve_knowledge': np.mean(results['eve_knowledge_values']),
            'std_eve_knowledge': np.std(results['eve_knowledge_values']),
            'qber_values': results['qber_values'],
            'eve_knowledge_values': results['eve_knowledge_values']
        }


def visualize_results(simulation_results: dict):
    """Create visualization of QBER and Eve's knowledge"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # QBER Distribution
    ax1.hist(simulation_results['qber_values'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(simulation_results['mean_qber'], color='red', linestyle='--', 
                label=f'Mean QBER: {simulation_results["mean_qber"]:.4f}')
    ax1.axvline(0.0612, color='green', linestyle='--', 
                label='Theoretical: 6.12%')
    ax1.axvline(0.11, color='orange', linestyle='--', 
                label='11% Threshold')
    ax1.set_xlabel('QBER')
    ax1.set_ylabel('Frequency')
    ax1.set_title('QBER Distribution with Helstrom Bound Attack')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Eve's Knowledge Distribution
    ax2.hist(simulation_results['eve_knowledge_values'], bins=30, alpha=0.7, 
             color='purple', edgecolor='black')
    ax2.axvline(simulation_results['mean_eve_knowledge'], color='red', linestyle='--',
                label=f'Mean: {simulation_results["mean_eve_knowledge"]:.4f}')
    ax2.axvline(0.938, color='green', linestyle='--',
                label='Theoretical: 93.8%')
    ax2.set_xlabel("Eve's Knowledge (%)")
    ax2.set_ylabel('Frequency')
    ax2.set_title("Eve's Knowledge of Bob's Bits After Basis Announcement")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def calculate_error_correction_leakage(qber: float, fc: float = 1.1) -> float:
    """
    Calculate error correction leakage: QBER * Fc
    """
    return qber * fc

def calculate_eve_ec_advantage(qber: float, fc: float = 1.1) -> float:
    """
    Calculate Eve's additional advantage from error correction: QBER * (Fc - 1)
    This is the extra information Eve gains beyond what she already knew from causing QBER
    """
    return qber * (fc - 1)


def plot_measurement_error_analysis(error_range: Tuple[float, float] = (0.0, 0.25), 
                                   num_points: int = 26, num_qubits: int = 1000000):
    """Plot QBER, Eve's knowledge, and Alice's knowledge vs measurement error rate with sampling rate analysis"""
    error_rates = np.linspace(error_range[0], error_range[1], num_points)
    
    # First row data (100% sampling)
    results = {
        'error_rates': [],
        'qber_values': [],
        'eve_knowledge_values': [],
        'alice_knowledge_values': []
    }
    
    print(f"Running simulations for measurement errors from {error_range[0]:.2%} to {error_range[1]:.2%}...")
    
    for error_rate in error_rates:
        print(f"Simulating with {error_rate:.2%} measurement error...")
        sim = BB84Simulation(num_qubits=num_qubits, measurement_error=error_rate)
        result = sim.run_protocol()
        
        results['error_rates'].append(error_rate)
        results['qber_values'].append(result['qber'])
        results['eve_knowledge_values'].append(result['eve_knowledge_percentage'])
        results['alice_knowledge_values'].append(result['alice_knowledge_percentage'])
    
    # Second row data (different sampling rates)
    sampling_rates = [0.85, 0.9, 0.95]
    all_sampling_results = {}
    
    for sampling_rate in sampling_rates:
        print(f"Running simulations for {sampling_rate:.0%} Eve sampling rate...")
        
        # For sampling rate analysis, start from Helstrom bound (9.2% measurement error)
        sampling_error_rates = np.linspace(0.092, error_range[1], num_points)
        
        sampling_results = {
            'qber_values': [],
            'eve_knowledge_values': [],
            'alice_knowledge_values': []
        }
        
        for error_rate in sampling_error_rates:
            sim = BB84Simulation(num_qubits=num_qubits, measurement_error=error_rate, 
                               eve_sampling_rate=sampling_rate)
            result = sim.run_protocol()
            
            qber = result['qber']
            alice_knowledge = result['alice_knowledge_percentage']
            eve_knowledge = result['eve_knowledge_percentage']
            
            # Calculate net knowledge after error correction leakage
            ec_leakage = calculate_error_correction_leakage(qber)
            eve_ec_advantage = calculate_eve_ec_advantage(qber)
            
            # Alice loses the full leakage, Eve gains only the additional advantage
            net_alice_knowledge = alice_knowledge  # Alice doesn't lose knowledge from EC
            net_eve_knowledge = eve_knowledge + eve_ec_advantage
            
            sampling_results['qber_values'].append(qber)
            sampling_results['eve_knowledge_values'].append(eve_knowledge)
            sampling_results['alice_knowledge_values'].append(alice_knowledge)
            sampling_results['net_alice_knowledge_values'] = sampling_results.get('net_alice_knowledge_values', [])
            sampling_results['net_eve_knowledge_values'] = sampling_results.get('net_eve_knowledge_values', [])
            sampling_results['net_alice_knowledge_values'].append(net_alice_knowledge)
            sampling_results['net_eve_knowledge_values'].append(net_eve_knowledge)
        
        all_sampling_results[sampling_rate] = sampling_results
    
    # Create 3x3 subplot layout
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes_top = axes[0]     # First row: original analysis
    axes_middle = axes[1]  # Second row: knowledge vs QBER
    axes_bottom = axes[2]  # Third row: net knowledge after error correction
    
    # First row plots
    # QBER vs Measurement Error
    axes_top[0].plot(np.array(results['error_rates']) * 100, np.array(results['qber_values']) * 100, 
                     marker='o', linewidth=2, markersize=4)
    axes_top[0].axhline(6.12, color='green', linestyle='--', label='Helstrom Theoretical: 6.12%')
    axes_top[0].axhline(11, color='red', linestyle='--', label='11% Security Threshold')
    axes_top[0].set_xlabel('Measurement Error Rate (%)')
    axes_top[0].set_ylabel('QBER (%)')
    axes_top[0].set_title('QBER vs Measurement Error Rate')
    axes_top[0].set_xlim(0, 25)
    axes_top[0].legend()
    axes_top[0].grid(True, alpha=0.3)
    
    # Knowledge vs Measurement Error
    axes_top[1].plot(np.array(results['error_rates']) * 100, np.array(results['eve_knowledge_values']) * 100,
                     marker='o', linewidth=2, markersize=4, color='purple', label="Eve's Knowledge")
    axes_top[1].plot(np.array(results['error_rates']) * 100, np.array(results['alice_knowledge_values']) * 100,
                     marker='s', linewidth=2, markersize=4, color='orange', label="Alice's Knowledge")
    axes_top[1].set_xlabel('Measurement Error Rate (%)')
    axes_top[1].set_ylabel('Knowledge (%)')
    axes_top[1].set_title("Knowledge vs Measurement Error Rate")
    axes_top[1].set_xlim(0, 25)
    axes_top[1].legend()
    axes_top[1].grid(True, alpha=0.3)
    
    # Knowledge vs QBER
    axes_top[2].plot(np.array(results['qber_values']) * 100, np.array(results['eve_knowledge_values']) * 100,
                     marker='o', linewidth=2, markersize=4, color='purple', label="Eve's Knowledge")
    axes_top[2].plot(np.array(results['qber_values']) * 100, np.array(results['alice_knowledge_values']) * 100,
                     marker='s', linewidth=2, markersize=4, color='orange', label="Alice's Knowledge")
    axes_top[2].axvline(6.12, color='green', linestyle=':', alpha=0.7, label='Helstrom QBER: 6.12%')
    axes_top[2].axvline(11, color='red', linestyle=':', alpha=0.7, label='11% Security Threshold')
    axes_top[2].set_xlabel('QBER (%)')
    axes_top[2].set_ylabel('Knowledge (%)')
    axes_top[2].set_title("Knowledge vs QBER")
    axes_top[2].set_xlim(0, None)
    axes_top[2].legend()
    axes_top[2].grid(True, alpha=0.3)
    
    # Second row plots (knowledge vs QBER)
    for i, sampling_rate in enumerate(sampling_rates):
        ax = axes_middle[i]
        sampling_results = all_sampling_results[sampling_rate]
        
        qber_vals = np.array(sampling_results['qber_values']) * 100
        eve_vals = np.array(sampling_results['eve_knowledge_values']) * 100
        alice_vals = np.array(sampling_results['alice_knowledge_values']) * 100
        
        # Plot knowledge vs QBER
        ax.plot(qber_vals, eve_vals, marker='o', linewidth=2, markersize=4, color='purple', label="Eve's Knowledge")
        ax.plot(qber_vals, alice_vals, marker='s', linewidth=2, markersize=4, color='orange', label="Alice's Knowledge")
        
        # Label the starting point (first data point)
        if len(qber_vals) > 0:
            start_qber = qber_vals[0]
            start_alice = alice_vals[0]
            start_eve = eve_vals[0]
            
            # Add annotation for the starting point
            ax.annotate(f'QBER: {start_qber:.1f}%\nAlice: {start_alice:.1f}%\nEve: {start_eve:.1f}%',
                       xy=(start_qber, start_alice), xytext=(10, 10),
                       textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.axvline(6.12, color='green', linestyle=':', alpha=0.7, label='Helstrom QBER: 6.12%')
        ax.axvline(11, color='red', linestyle=':', alpha=0.7, label='11% Security Threshold')
        
        ax.set_xlabel('QBER (%)')
        ax.set_ylabel('Knowledge (%)')
        ax.set_title(f"Knowledge vs QBER ({sampling_rate:.0%} Sampling)")
        
        # Set x-axis to start from minimum QBER value and stop at 12%
        min_qber = min(qber_vals)
        ax.set_xlim(min_qber * 0.95, 12)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Third row plots (net knowledge after error correction)
    for i, sampling_rate in enumerate(sampling_rates):
        ax = axes_bottom[i]
        sampling_results = all_sampling_results[sampling_rate]
        
        qber_vals = np.array(sampling_results['qber_values']) * 100
        net_eve_vals = np.array(sampling_results['net_eve_knowledge_values']) * 100
        net_alice_vals = np.array(sampling_results['net_alice_knowledge_values']) * 100
        
        # Plot net knowledge vs QBER (after error correction leakage)
        ax.plot(qber_vals, net_eve_vals, marker='o', linewidth=2, markersize=4, color='purple', label="Eve's Net Knowledge")
        
        # Label the starting point (first data point)
        if len(qber_vals) > 0:
            start_qber = qber_vals[0]
            start_alice = net_alice_vals[0]
            start_eve = net_eve_vals[0]
            
            # Add annotation for the starting point
            ax.annotate(f'QBER: {start_qber:.1f}%\nAlice: {start_alice:.1f}%\nEve: {start_eve:.1f}%',
                       xy=(start_qber, start_alice), xytext=(10, 10),
                       textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        ax.plot(qber_vals, net_alice_vals, marker='s', linewidth=2, markersize=4, color='orange', label="Alice's Net Knowledge")
        
        ax.axvline(6.12, color='green', linestyle=':', alpha=0.7, label='Helstrom QBER: 6.12%')
        ax.axvline(11, color='red', linestyle=':', alpha=0.7, label='11% Security Threshold')
        
        ax.set_xlabel('QBER (%)')
        ax.set_ylabel('Net Knowledge (%) - After EC')
        ax.set_title(f"Net Knowledge after EC ({sampling_rate:.0%} Sampling, Fc=1.1)")
        
        # Set x-axis to start from minimum QBER value and stop at 12%
        min_qber = min(qber_vals)
        ax.set_xlim(min_qber * 0.95, 12)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return results, all_sampling_results, fig



# Main demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("BB84 QKD Measurement Error Analysis")
    print("=" * 60)
    
    # Run single simulation with Helstrom bound
    print("\n1. Running single simulation with Helstrom bound (9.2% error)...")
    sim = BB84Simulation(num_qubits=1000000)
    result = sim.run_protocol()
    
    print(f"\nHelstrom Bound Results:")
    print(f"  - QBER: {result['qber']:.4%}")
    print(f"  - Eve's Knowledge: {result['eve_knowledge_percentage']:.4%}")
    print(f"  - Alice's Knowledge: {result['alice_knowledge_percentage']:.4%}")
    print(f"  - Sifted Key Length: {result['sifted_key_length']}")
    
    # Plot measurement error analysis and sampling rate analysis from 0% to 25%
    print("\n2. Analyzing QBER and Eve's knowledge vs measurement error (0% to 25%) with sampling rate analysis...")
    results, sampling_results, fig = plot_measurement_error_analysis(error_range=(0.0, 0.25), num_points=26)
    plt.show()
    
    print(f"\nError Analysis Summary:")
    print(f"  - Min QBER: {min(results['qber_values']):.4%} at {results['error_rates'][np.argmin(results['qber_values'])]:.1%} error")
    print(f"  - Max QBER: {max(results['qber_values']):.4%} at {results['error_rates'][np.argmax(results['qber_values'])]:.1%} error")
    print(f"  - Min Eve Knowledge: {min(results['eve_knowledge_values']):.4%} at {results['error_rates'][np.argmin(results['eve_knowledge_values'])]:.1%} error")
    print(f"  - Max Eve Knowledge: {max(results['eve_knowledge_values']):.4%} at {results['error_rates'][np.argmax(results['eve_knowledge_values'])]:.1%} error")
    print(f"  - Min Alice Knowledge: {min(results['alice_knowledge_values']):.4%} at {results['error_rates'][np.argmin(results['alice_knowledge_values'])]:.1%} error")
    print(f"  - Max Alice Knowledge: {max(results['alice_knowledge_values']):.4%} at {results['error_rates'][np.argmax(results['alice_knowledge_values'])]:.1%} error")
    
    print(f"\nSampling Rate Analysis:")
    for rate in [0.85, 0.9, 0.95]:
        sr = sampling_results[rate]
        print(f"  {rate:.0%} Sampling - Max Eve Knowledge: {max(sr['eve_knowledge_values']):.4%}, Min QBER: {min(sr['qber_values']):.4%}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
