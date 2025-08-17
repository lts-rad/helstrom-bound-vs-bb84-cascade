#https://wiki.veriqloud.fr/index.php?title=BB84_Quantum_Key_Distribution
# TBD finish verifying analysis here questions remain for how to 
# properly model QBER vs cascade leakage with quadrature measurements
# this is a worst case bound and the real cascade implementation 
# is unlikely to be leaking 100% of the info at these boundaries, perhaps closer
# to 40% of unknown bit flips in sim

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from helstrom import *

def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def calculate_asymptotic_key_rate(amplitude, eve_sampling_rate, gamma=0.5, f_EC=1.0):
    alpha_squared = amplitude**2
    helstrom_error = helstrom_bound_4psk(alpha_squared)

    if helstrom_error is None:
        return 0, 0

    base_qber = (2/3) * helstrom_error
    eve_knowledge = 1 - base_qber

    # For coherent attack with sampling
    effective_QBER = base_qber * eve_sampling_rate
    error_correction_cost = f_EC * binary_entropy(effective_QBER)
    eve_effective_knowledge = eve_knowledge * eve_sampling_rate
    privacy_amplification_cost = eve_effective_knowledge

    total_leakage_rate = error_correction_cost + privacy_amplification_cost
    asymptotic_key_rate = (1 - gamma)**2 * (1 - total_leakage_rate)

    return effective_QBER, asymptotic_key_rate

# Parameters
gamma = 0.5
f_EC = 1.1  # Cascade protocol inefficiency
n = 1e9  # Fixed key length

# Create ranges
amplitude_range = np.linspace(0.5, np.sqrt(2), 100)  # Extended to sqrt(2) ≈ 1.414
sampling_rates = np.linspace(0.5, 1.0, 100)  # 100 points for sampling rate

print("=== Simple 3D Analysis: QBER vs Amplitude vs Asymptotic Key Rate ===")
print(f"Key length: {n:.0e}")
print(f"Sampling rate gradient: {sampling_rates[0]:.1f} to {sampling_rates[-1]:.1f}")

plt.style.use('default')
fig = plt.figure(figsize=(12, 8), dpi=150)
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for the surface
Amplitude_mesh, SamplingRate_mesh = np.meshgrid(amplitude_range, sampling_rates)
QBER_mesh = np.zeros_like(Amplitude_mesh)
KeyRate_mesh = np.zeros_like(Amplitude_mesh)

print("Computing gradient surface...")
# Calculate data for the gradient surface
for i, sampling_rate in enumerate(sampling_rates):
    for j, amp in enumerate(amplitude_range):
        effective_qber, key_rate = calculate_asymptotic_key_rate(amp, sampling_rate)
        QBER_mesh[i, j] = effective_qber * 100  # Convert to percentage
        KeyRate_mesh[i, j] = key_rate

# Filter data to limit QBER to 15%
mask = QBER_mesh <= 15  # Now in percentage
QBER_filtered = np.where(mask, QBER_mesh, np.nan)
KeyRate_filtered = np.where(mask, KeyRate_mesh, np.nan)
SamplingRate_filtered = np.where(mask, SamplingRate_mesh, np.nan)

# Create the surface with new axis arrangement: QBER vs Amplitude vs Sampling Rate
# Color represents the asymptotic key rate
import matplotlib.colors as mcolors

# Create custom colormap for key rates: yellow at 0, red at most negative
colors_keyrate = ['#8B0000', '#FF4500', '#FFD700']  # Dark red (most negative) -> Orange -> Yellow (at 0)
cmap_keyrate = mcolors.LinearSegmentedColormap.from_list('keyrate', colors_keyrate, N=256)

# Create face colors based on key rate values
face_colors = np.zeros(KeyRate_filtered.shape + (4,))  # RGBA

for i in range(KeyRate_filtered.shape[0]):
    for j in range(KeyRate_filtered.shape[1]):
        if not np.isnan(KeyRate_filtered[i, j]):
            keyrate_val = KeyRate_filtered[i, j]
            # Check if amplitude > 1.0 OR QBER > 11% for purple coloring
            amp_val = amplitude_range[j]
            qber_val = QBER_mesh[i, j]

            if keyrate_val >= 0:
                # Solid gray for ALL positive key rates (secure)
                face_colors[i, j] = [0.7, 0.7, 0.7, 0.9]  # Solid gray
            else:
                # Check if amplitude > 1.0 OR QBER > 11% for purple vs yellow-red gradient
                if amp_val > 1.0 or qber_val > 11.0:
                    # Purple gradient for amplitudes > 1.0 OR QBER > 11%
                    # Map from light purple (at 0) to dark purple (most negative)
                    purple_intensity = abs(keyrate_val) / 0.15  # 0 to 1
                    purple_intensity = max(0, min(1, purple_intensity))
                    purple_color = [0.5 + 0.3*purple_intensity, 0.3*(1-purple_intensity), 0.5 + 0.3*purple_intensity, 0.9]
                    face_colors[i, j] = purple_color
                else:
                    # Yellow to red gradient for negative key rates (amplitude ≤ 1.0 AND QBER ≤ 11%)
                    normalized_val = (keyrate_val + 0.15) / 0.15  # Map [-0.15, 0] to [0, 1]
                    normalized_val = max(0, min(1, normalized_val))
                    rgba_color = cmap_keyrate(normalized_val)  # Yellow at 0, red at most negative
                    face_colors[i, j] = rgba_color

# Plot with sampling rate as z-axis
surf = ax.plot_surface(QBER_filtered, Amplitude_mesh, SamplingRate_mesh,
                       facecolors=face_colors,
                       alpha=0.9, linewidth=0, antialiased=True, shade=False,
                       rcount=100, ccount=100)  # Explicitly set resolution to match data

amp_range_11 = np.linspace(0.5, 1.0, 20)
sampling_range_11 = np.linspace(0.5, 1.0, 20)
AMP_11, SAMPLING_11 = np.meshgrid(amp_range_11, sampling_range_11)
qber_11_plane_x = np.full_like(AMP_11, 11)  # Vertical plane at 11% QBER

ax.plot_surface(qber_11_plane_x, AMP_11, SAMPLING_11,
                color='black', alpha=0.15, linewidth=0)

ax.text(11, 0.75, 0.9, '11%', fontsize=11, color='black', weight='bold')


# Create custom colorbar
cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height]

# Draw negative portion (yellow to red gradient)
# Need to flip the colormap for the colorbar to match surface
negative_values = np.linspace(0, 1, 100).reshape(-1, 1)  # 0 to 1 for colormap
cbar_ax.imshow(negative_values, aspect='auto', cmap=cmap_keyrate, extent=[0, 1, -0.15, 0], origin='lower')

# Draw positive portion (solid gray)
positive_rect = patches.Rectangle((0, 0), 1, 0.15, facecolor='gray', alpha=0.9)
cbar_ax.add_patch(positive_rect)

cbar_ax.set_ylabel('Asymptotic Key Rate', fontsize=11, labelpad=15)
cbar_ax.set_yticks([-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
cbar_ax.set_yticklabels(['-0.15', '-0.10', '-0.05', '0', '0.05', '0.10', '0.15'], fontsize=9)
cbar_ax.set_xlim(0, 1)
cbar_ax.set_ylim(-0.15, 0.15)
cbar_ax.set_xticks([])  # Remove x-axis ticks

# Add subtle border to colorbar
for spine in cbar_ax.spines.values():
    spine.set_edgecolor('lightgray')
    spine.set_linewidth(0.5)

ax.set_xlabel('Effective QBER (%)', fontsize=12, labelpad=10)
ax.set_ylabel('Amplitude α (Mean Photon Number = α²)', fontsize=12, labelpad=10)
ax.set_zlabel('Eve Sampling Rate', fontsize=12, labelpad=10)
ax.set_title('Asymptotic Key Rate vs QBER and Amplitude\nunder Coherent Attack',
             fontsize=16, pad=25, weight='bold')

# Set limits for better view (focus on critical region)
ax.set_xlim(15, 0)  # Reversed: 15% to 0% (higher security toward front)
ax.set_ylim(0.5, np.sqrt(2))  # Updated to match new amplitude range
ax.set_zlim(0.5, 1.0)  # Sampling rate range

ax.grid(True, alpha=0.2, linewidth=0.5)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Minimal pane edges
ax.xaxis.pane.set_edgecolor('lightgray')
ax.yaxis.pane.set_edgecolor('lightgray')
ax.zaxis.pane.set_edgecolor('lightgray')
ax.xaxis.pane.set_alpha(0.05)
ax.yaxis.pane.set_alpha(0.05)
ax.zaxis.pane.set_alpha(0.05)

ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.tick_params(axis='z', labelsize=10)

# Add custom y-axis tick labels showing both amplitude and mean photon number
amp_ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, np.sqrt(2)]
tick_labels = [f'{amp:.1f}\n({amp**2:.2f})' if amp != np.sqrt(2) else f'{amp:.2f}\n({amp**2:.1f})' for amp in amp_ticks]
ax.set_yticks(amp_ticks)
ax.set_yticklabels(tick_labels, fontsize=8)  # Smaller font to prevent overlap

sampling_ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ax.set_zticks(sampling_ticks)
ax.set_zticklabels([f'{s:.1f}' for s in sampling_ticks])

ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()

