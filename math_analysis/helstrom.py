#Helstrom Bound POVM Calculations for 4-QPSK

#Calculation taken from  https://www.nature.com/articles/s41598-019-55589-7 equations 54, 56 for N=4

import numpy as np
def helstrom_bound_4psk(alpha_squared):
  """
  Computes the Helstrom bound for 4-PSK coherent states.
  The Helstrom bound gives the minimum average probability of error
  for discriminating between a set of quantum states. For N-PSK
  with equal prior probabilities, it's given by:
  P(Hel) = 1 - (1/N^2) * Σ[sqrt(h_p)]^2 for p=1 to N
  where h_p are the eigenvalues of a related Gram matrix.
  For 4-PSK (N=4), the eigenvalues h_p are given by:
  h₁ = 2 * exp(-α²) * (cosh(α²) + cos(α²))
  h₂ = 2 * exp(-α²) * (sinh(α²) + sin(α²))
  h₃ = 2 * exp(-α²) * (cosh(α²) - cos(α²))
  h₄ = 2 * exp(-α²) * (sinh(α²) - sin(α²))
  Args:
    alpha_squared: The square of the coherent state amplitude (α²),
                   often representing the average photon number (n_s).
                   Must be non-negative.
  Returns:
    The Helstrom bound (minimum error probability) for 4-PSK,
    or None if input is invalid.
  """
  if alpha_squared < 0:
    print("Error: alpha_squared (average photon number) cannot be negative.")
    return None
  N = 4
  exp_minus_alpha_sq = np.exp(-alpha_squared)
  # Calculate the eigenvalues h_p for N=4
  h1 = 2 * exp_minus_alpha_sq * (np.cosh(alpha_squared) + np.cos(alpha_squared))
  h2 = 2 * exp_minus_alpha_sq * (np.sinh(alpha_squared) + np.sin(alpha_squared))
  h3 = 2 * exp_minus_alpha_sq * (np.cosh(alpha_squared) - np.cos(alpha_squared))
  # Ensure h4 is non-negative, as sinh(x) can be < sin(x) for small negative x,
  # although alpha_squared should be >= 0. Taking max with 0 for safety.
  h4 = 2 * exp_minus_alpha_sq * (np.sinh(alpha_squared) - np.sin(alpha_squared))
  h4 = max(0, h4) # sinh(x) >= sin(x) for x >= 0, so h4 >= 0 if alpha_squared >= 0
  # Calculate the sum of the square roots of the eigenvalues
  # Ensure arguments to sqrt are non-negative
  sum_sqrt_h = np.sqrt(max(0, h1)) + np.sqrt(max(0, h2)) + np.sqrt(max(0, h3)) + np.sqrt(h4)
  # Calculate the Helstrom bound
  sum_term = (1.0 / N) * sum_sqrt_h
  p_hel = 1.0 - sum_term * sum_term
  # The bound should be between 0 and 1
  return max(0.0, min(p_hel, 1.0))

print(helstrom_bound_4psk(1))
print(helstrom_bound_4psk(0.9))
print(helstrom_bound_4psk(0.8))
print(helstrom_bound_4psk(2**0.5))

