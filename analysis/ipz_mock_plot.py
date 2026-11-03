import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ==========================================
# Define Time and Stimulus Parameters
# ==========================================
t = np.linspace(-0.5, 1.5, 2000)
t_on = 0.0
t_off = 0.25

tau_act = 0.015    # Fast activation
tau_inact = 0.4    # Slower inactivation, decaying to ~ -0.6 at t_off
tau_deact = 0.05   # Fast deactivation after offset

# ==========================================
# Generate Mock I_pz Data
# ==========================================
I_pz = np.zeros_like(t)

# Phase A & B: During stimulus (Activation + Inactivation)
mask_stim = (t >= t_on) & (t < t_off)
t_stim = t[mask_stim] - t_on

# Double exponential forced to peak at exactly -1.0
raw_on = np.exp(-t_stim / tau_inact) - np.exp(-t_stim / tau_act)
peak_val = np.max(raw_on)
I_pz[mask_stim] = -raw_on / peak_val

# Phase C: After stimulus (Deactivation)
mask_post = t >= t_off
t_post = t[mask_post] - t_off

# Start deactivation from the exact value reached before stimulus offset
I_last = I_pz[mask_stim][-1]
I_pz[mask_post] = I_last * np.exp(-t_post / tau_deact)

# Generate stimulus trace for the top panel
stimulus = np.zeros_like(t)
stimulus[(t >= t_on) & (t < t_off)] = 1.0

# ==========================================
# Plotting
# ==========================================
SECONDARY_COLOR = 'gray'

fig, (ax_stim, ax_ipz) = plt.subplots(
    2, 1, 
    figsize=(8, 6), 
    sharex=True, 
    gridspec_kw={'height_ratios': [1, 6]} 
)

# --- Top Panel: Stimulus ---
ax_stim.plot(t, stimulus, color=SECONDARY_COLOR, alpha=0.5, linewidth=1.5)

ax_stim.set_ylabel('Mech.\nstim.', rotation=0, labelpad=30, va='center', fontsize=12, color=SECONDARY_COLOR)
ax_stim.set_yticks([0, 1])
ax_stim.set_yticklabels(['off', 'on'], color=SECONDARY_COLOR, fontsize=10)

ax_stim.spines['top'].set_visible(False)
ax_stim.spines['right'].set_visible(False)
ax_stim.spines['bottom'].set_visible(False)
ax_stim.spines['left'].set_color(SECONDARY_COLOR)
ax_stim.tick_params(axis='x', length=0)
ax_stim.tick_params(axis='y', colors=SECONDARY_COLOR)

# --- Bottom Panel: I_pz Mock Data ---
ax_ipz.plot(t, I_pz, color='#FF5733', linewidth=2.5, label='Schematic $I_{pz}$')

# Styling matched to plot_spike_details
ax_ipz.set_ylabel(r'Normalized $I_{pz}$', labelpad=30, va='center', fontsize=14)
ax_ipz.set_xlabel('Relative time in a.u.', fontsize=14)
ax_ipz.tick_params(axis='both', labelsize=12)

ax_ipz.spines['top'].set_visible(False)
ax_ipz.spines['right'].set_visible(False)

ax_ipz.set_xlim(-0.5, 1.5)
ax_ipz.set_ylim(-1.05, 0.1)

# --- Annotate the Three Kinetic Phases ---
# 1. Activation (horizontal arrow pointing right)
ax_ipz.annotate(r'$\mathbf{Activation}$' + '\n(opens rapidly)', 
             xy=(0.012, -0.5), xytext=(-0.3, -0.5),
             arrowprops=dict(color='black', arrowstyle='->', lw=1.2, shrinkA=8), 
             fontsize=11, ha='center', va='center')

# 2. Inactivation (diagonal arrow pointing up-left)
ax_ipz.annotate(r'$\mathbf{Inactivation}$' + '\n(slow decay during stimulus)', 
             xy=(0.14, -0.8), xytext=(0.75, -0.9),
             arrowprops=dict(color='black', arrowstyle='->', lw=1.2, shrinkA=8), 
             fontsize=11, ha='center', va='center')

# 3. Deactivation (horizontal arrow pointing left)
ax_ipz.annotate(r'$\mathbf{Deactivation}$' + '\n(fast closure upon offset)', 
             xy=(0.28, -0.35), xytext=(0.85, -0.35),
             arrowprops=dict(color='black', arrowstyle='->', lw=1.2, shrinkA=8), 
             fontsize=11, ha='center', va='center')

ax_ipz.legend(loc='lower right', frameon=False)

fig.suptitle('Schematic of Piezo1 Kinetic Phases', fontsize=14)

plt.tight_layout()
plt.savefig("output/mock_ipz_kinetic_phases.png", dpi=300, bbox_inches="tight")
plt.savefig("output/mock_ipz_kinetic_phases.svg", format="svg", bbox_inches="tight", transparent=True)
plt.close()
