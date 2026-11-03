import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

PRIMARY_COLOR = "#b41f1f"  # Red for piezo current
SECOND_PRIMARY_COLOR = "#151FA1"  # Blue for membrane potential
SECONDARY_COLOR = "#747474"  # Gray for mechanical stimulus

# Update global font sizes
plt.rcParams.update({
    'font.size': 14,          # Global font size
    'axes.labelsize': 16,     # X and Y axis labels
    'xtick.labelsize': 14,    # X axis tick labels (the numbers)
    'ytick.labelsize': 14,    # Y axis tick labels
    'legend.fontsize': 12,    # Legend text
    'axes.titlesize': 18      # Plot title
})

def plot_simple_time_series(t, y, ylabel, filepath, log_x=False, log_y=False):
    """
    Creates a simple time-series plot matching the project's grid and spine style.
    
    Parameters:
    t (array-like): Time array (x-axis).
    y (array-like): Data array (y-axis).
    ylabel (str): Label for the y-axis.
    filepath (str): Path to save the figure.
    log_x (bool): If True, sets the x-axis to a logarithmic scale.
    log_y (bool): If True, sets the y-axis to a logarithmic scale.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot the data using the globally defined PRIMARY_COLOR
    ax.plot(t, y, color=PRIMARY_COLOR, linewidth=1.5)
    
    # Set titles and labels
    ax.set_xlabel('Time in s', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Apply logarithmic scaling if requested
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
        
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_dual_time_series(t, y1, y2, ylabel1, ylabel2, filepath, 
                            log_x=False, log_y1=False, log_y2=False, 
                            label1="Primary Trace", label2="Secondary Trace"):
    """
    Creates a time-series plot with two y-axes (left and right).
    
    Parameters:
    t (array-like): Time array (x-axis).
    y1 (array-like): Data for the left y-axis.
    y2 (array-like): Data for the right y-axis.
    ylabel1 (str): Label for the left y-axis.
    ylabel2 (str): Label for the right y-axis.
    filepath (str): Path to save the figure.
    log_x (bool): If True, sets the x-axis to a logarithmic scale.
    log_y1 (bool): If True, sets the left y-axis to a logarithmic scale.
    log_y2 (bool): If True, sets the right y-axis to a logarithmic scale.
    label1 (str): Legend label for the primary trace.
    label2 (str): Legend label for the secondary trace.
    """
    fig, ax1 = plt.subplots(figsize=(10, 4))
    
    # Plot Primary Data (Left Axis)
    line1 = ax1.plot(t, y1, color=PRIMARY_COLOR, linewidth=1.5, label=label1)
    
    ax1.set_xlabel('Time in s', fontsize=12)
    ax1.set_ylabel(ylabel1, color=PRIMARY_COLOR, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=PRIMARY_COLOR)
    
    # Plot Secondary Data (Right Axis)
    ax2 = ax1.twinx()  # Create a second axes that shares the same x-axis
    
    # Using a dashed line for the secondary trace to clearly distinguish it
    line2 = ax2.plot(t, y2, color=SECOND_PRIMARY_COLOR, linewidth=1.5, linestyle='--', label=label2)
    
    ax2.set_ylabel(ylabel2, color=SECOND_PRIMARY_COLOR, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=SECOND_PRIMARY_COLOR)
    
    # Apply Logarithmic Scaling if requested
    if log_x:
        ax1.set_xscale('log')
    if log_y1:
        ax1.set_yscale('log')
    if log_y2:
        ax2.set_yscale('log')
        
    # Unified Legend
    # Since we have two different axes, we need to gather the lines and labels from both
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=10, framealpha=1.0)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def plot_piezo_current_peaks(t1, ipz, t2, mech, glut, ipz_label, ipz_limits, mech_label, mech_limits, output_folder, plot_title):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Main component: I_pz
    ax1.plot(t1, ipz, color=PRIMARY_COLOR, label='$I_{pz}$')
    ax1.set_xlabel('Time in s', fontsize=12)
    ax1.set_ylabel(ipz_label, fontsize=12, color=PRIMARY_COLOR)
    ax1.tick_params(axis='y', labelcolor=PRIMARY_COLOR)
    ax1.set_ylim(ipz_limits[0], ipz_limits[1]) 

    # Additional component: Mechanical Stimulus
    ax2 = ax1.twinx()
    ax2.plot(t2, mech, color=SECONDARY_COLOR, linestyle='--', alpha=0.8, label='Mech. stimulus')
    ax2.set_ylabel(mech_label, fontsize=12, color=SECONDARY_COLOR)
    ax2.tick_params(axis='y', labelcolor=SECONDARY_COLOR)
    ax2.set_ylim(mech_limits[0], mech_limits[1]) 

    # Glutamate states as background shading
    ax1.fill_between(t2, 0, 1, where=np.isclose(glut, 0.001, atol=1e-4),
                     color='lightgray', alpha=0.3, transform=ax1.get_xaxis_transform(),
                     label='[glu] = 0.001 M')

    ax1.fill_between(t2, 0, 1, where=np.isclose(glut, 0.006, atol=1e-4),
                     color='gray', alpha=0.4, transform=ax1.get_xaxis_transform(),
                     label='[glu] = 0.006 M')

    # Legend and Formatting (4 items in a 2x2 grid below the plot)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    # Order for column-major filling in Matplotlib (Col 1: signals, Col 2: glutamate states):
    # [I_pz, Mech. stimulus, [glu] = 0.001 M, [glu] = 0.006 M]
    all_handles = [lines_1[0], lines_2[0], lines_1[1], lines_1[2]]
    all_labels = [labels_1[0], labels_2[0], labels_1[1], labels_1[2]]

    ax1.legend(
        all_handles, 
        all_labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.3), 
        ncol=2, 
        framealpha=0.9
    )

    plt.title(plot_title, fontsize=14, pad=15)

    fig.tight_layout()
    # plt.savefig(f"{output_folder}/fig1.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_folder}/fig1.svg", format="svg", bbox_inches="tight", transparent=True)
    plt.close()


def plot_spike_details(t1, ipz, t2, mech, mech_label, mech_unit, filepath, plot_title, group_idx=0):
    """
    Extracts and overlays normalized responses for one of 3 stimulus groups,
    with 'on'/'off' state labels for the mechanical stimulus. Baseline drift 
    is removed prior to normalization.
    """
    # Ensure inputs are 1D arrays
    mech = np.asarray(mech).squeeze()
    t2 = np.asarray(t2).squeeze()
    ipz = np.asarray(ipz).squeeze()
    t1 = np.asarray(t1).squeeze()
    
    # Detect all mechanical stimuli in the full simulation
    threshold = 0.05 * np.max(mech) if np.max(mech) > 0 else 0
    is_on = (mech > threshold).astype(int)
    
    diff = np.diff(is_on)
    onsets = np.where(diff == 1)[0] + 1
    offsets = np.where(diff == -1)[0] + 1
    
    if is_on[0] == 1:
        onsets = np.insert(onsets, 0, 0)
    if is_on[-1] == 1:
        offsets = np.append(offsets, len(mech) - 1)
        
    num_stimuli = len(onsets)
    if num_stimuli == 0:
        print("No mechanical stimuli detected.")
        return
        
    # Select the requested group of spikes
    stims_per_group = max(1, num_stimuli // 3)
    start_idx = group_idx * stims_per_group
    end_idx = min((group_idx + 1) * stims_per_group, num_stimuli)
    
    group_onsets = onsets[start_idx:end_idx]
    group_offsets = offsets[start_idx:end_idx]
    
    # Figure Setup:
    fig, (ax_stim, ax_ipz) = plt.subplots(
        2, 1, 
        figsize=(8, 6), 
        sharex=True, 
        gridspec_kw={'height_ratios': [1, 6]} 
    )
    
    intensities = [np.max(mech[on:off]) for on, off in zip(group_onsets, group_offsets)]
    cmap = plt.get_cmap('jet') 
    norm = mcolors.Normalize(vmin=np.min(intensities), vmax=np.max(intensities))
    
    max_duration = 0

    linestyles = ["solid", "dashed", "dotted"]
    
    # Extract, Normalize, and Overlay each stimulus
    for i, (onset_idx, offset_idx) in enumerate(zip(group_onsets, group_offsets)):
        t_onset = t2[onset_idx]
        t_offset = t2[offset_idx]
        
        duration = t_offset - t_onset
        if duration > max_duration:
            max_duration = duration
            
        # --- Top Panel ---
        mask_t2 = (t2 >= t_onset - 0.5) & (t2 <= t_offset + 1.25)
        t2_rel = t2[mask_t2] - t_onset
        mech_snippet = mech[mask_t2]
        
        stim_on_off = (mech_snippet > threshold).astype(float)
        label_stim = 'Stimulus' if i == 0 else None
        ax_stim.plot(t2_rel, stim_on_off, color=SECONDARY_COLOR, alpha=0.25, label=label_stim)
        
        # --- Bottom Panel ---
        mask_t1 = (t1 >= t_onset - 0.5) & (t1 <= t_offset + 1.25)
        t1_rel = t1[mask_t1] - t_onset
        ipz_snippet = ipz[mask_t1]
        
        # --- NEW: Baseline Correction ---
        # Calculate the baseline using the pre-stimulus window (relative time < 0)
        pre_stim_mask = t1_rel < 0
        if np.any(pre_stim_mask):
            baseline = np.mean(ipz_snippet[pre_stim_mask])
        else:
            baseline = ipz_snippet[0] # Fallback if no pre-stimulus data exists
            
        # Subtract baseline so all traces start at 0
        ipz_corrected = ipz_snippet - baseline
        
        # Normalize using the absolute peak of the baseline-corrected signal
        peak_val = np.abs(np.min(ipz_corrected))
        ipz_norm = ipz_corrected / peak_val if peak_val > 0 else ipz_corrected
            
        color = cmap(norm(intensities[i]))
        ax_ipz.plot(t1_rel, ipz_norm, color=color, linestyle=linestyles[i], label=f'{intensities[i]:g} {mech_unit}')
        
    # Styling
    ax_stim.set_ylabel('Mech.\nstim.', rotation=0, labelpad=30, va='center', fontsize=12, color=SECONDARY_COLOR)
    
    # Set y-ticks to show 'off' and 'on' states
    ax_stim.set_yticks([0, 1])
    ax_stim.set_yticklabels(['off', 'on'], color=SECONDARY_COLOR, fontsize=10)
    
    ax_stim.spines['top'].set_visible(False)
    ax_stim.spines['right'].set_visible(False)
    ax_stim.spines['bottom'].set_visible(False)
    ax_stim.spines['left'].set_color(SECONDARY_COLOR)
    ax_stim.tick_params(axis='x', length=0)
    ax_stim.tick_params(axis='y', colors=SECONDARY_COLOR) 
    
    # Bottom panel styles
    ax_ipz.set_ylabel(r'Normalized $I_{pz}$', labelpad=30, va='center', fontsize=14)
    ax_ipz.set_xlabel('Relative time in s', fontsize=14)
    
    ax_ipz.spines['top'].set_visible(False)
    ax_ipz.spines['right'].set_visible(False)
    
    ax_ipz.set_xlim(-0.5, max_duration + 1.25)
    
    # Discrete Legend
    ax_ipz.legend(loc='lower right', frameon=False, title=mech_label)

    fig.suptitle(plot_title, fontsize=14)
    
    plt.tight_layout()
    # plt.savefig(filepath, dpi=300)
    plt.savefig(filepath, format="svg", bbox_inches="tight", transparent=True)
    plt.close()


def plot_tau_comparison(data_dict, filepath, use_log_y=False, jitter_width=0.2, ylabel=r'Time Constant $\tau$ in s', start_0=False):
    """
    Plots time constants (tau) across different glutamate states and models.
    Applies a two-tiered horizontal jitter to separate models AND intensities.
    Can also be used for rise-time values instead of tau.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Guarantee space on the right side for the legend
    fig.subplots_adjust(right=0.75)
    
    models = list(data_dict.keys())
    markers = ['o', 's', '^', 'X', 'D'] 
    marker_map = {model: markers[i % len(markers)] for i, model in enumerate(models)}
    
    # Calculate primary horizontal offsets for models
    n_models = len(models)
    model_offsets = np.linspace(-jitter_width, jitter_width, n_models) if n_models > 1 else [0]
    offset_map = {model: model_offsets[i] for i, model in enumerate(models)}
    
    # Calculate secondary horizontal sub-offsets for intensities (to prevent overlap)
    # This slightly spreads the Low, Medium, and High dots within their model's cluster
    intensity_sub_offsets = np.linspace(-0.05, 0.05, 3)
    
    # Define the grouped colors
    rank_colors = ['navy', 'lightgreen', 'darkred']
    rank_labels = ['Low', 'Medium', 'High']
    
    # Extract unique glut states for the X-axis
    all_glut_states = set()
    for model in models:
        for glut_state in data_dict[model].keys():
            all_glut_states.add(glut_state)
            
    glut_states = sorted(list(all_glut_states))
    x_map = {state: i for i, state in enumerate(glut_states)} 
    
    # Plot the data
    for model in models:
        model_ints = set()
        for glut_state, (ints, taus) in data_dict[model].items():
            model_ints.update(ints)
        
        sorted_model_ints = sorted(list(model_ints))
        int_to_rank = {val: i for i, val in enumerate(sorted_model_ints)}
        
        for glut_state, (ints, taus) in data_dict[model].items():
            for intensity, tau in zip(ints, taus):
                if np.isnan(tau) or (use_log_y and tau <= 0):
                    continue
                    
                rank_idx = int_to_rank[intensity]
                rank_idx = min(rank_idx, len(rank_colors) - 1) 
                color = rank_colors[rank_idx]
                
                # Apply two-tiered jitter: Base pos + Model offset + Intensity sub-offset
                x_pos = x_map[glut_state] + offset_map[model] + intensity_sub_offsets[rank_idx]
                
                ax.scatter(x_pos, tau, 
                           color=color, 
                           marker=marker_map[model], 
                           s=120,             
                           alpha=0.85,         
                           edgecolors='black',
                           linewidth=1)

    # Styling
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel('Extracellular glutamate concentration', fontsize=14)
    
    ax.set_xticks(range(len(glut_states)))
    ax.set_xticklabels([str(g) for g in glut_states], fontsize=12)
    
    if use_log_y:
        ax.set_yscale('log')
    elif start_0:
        ax.set_ylim(bottom=0)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Create a Single, Unified Legend
    legend_elements = []
    
    # Section A: Models (Shapes)
    legend_elements.append(Line2D([0], [0], color='w', label=r'$\bf{Model\ (shape)}$'))
    for m in models:
        display_name = m.value if hasattr(m, 'value') else str(m)
        legend_elements.append(Line2D([0], [0], marker=marker_map[m], color='w', label=f'  {display_name}',
                                      markerfacecolor='gray', markersize=10, markeredgecolor='black'))
        
    # Spacer
    legend_elements.append(Line2D([0], [0], color='w', label=' '))
    
    # Section B: Intensities (Colors) - Label updated for clarity
    legend_elements.append(Line2D([0], [0], color='w', label=r'$\bf{Mech.\ stim.}$' + '\n' + r'$\bf{intensity\ (color)}$'))
    for i, color in enumerate(rank_colors):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'  {rank_labels[i]}',
                                      markerfacecolor=color, markersize=10, markeredgecolor='black'))
    
    # Place the unified legend outside the plot
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), 
              frameon=False, fontsize=11)
    
    # plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.savefig(filepath, format="svg", bbox_inches="tight", transparent=True)
    plt.close()


def plot_time_series_comparison(data_dict, dummy_key, filepath, ylabel):
    """
    Plots a generalized time series comparison for multiple models vs a dummy reference.
    Can be used for Membrane Potential (V_m), Intracellular Calcium (Ca_i), etc.
    
    Parameters:
    data_dict (dict): Dictionary where keys are model names (or Enums) and 
                      values are tuples of (time_array, y_array).
    dummy_key (str/Enum): The exact key in data_dict corresponding to the dummy model.
    filepath (str): Path to save the figure.
    ylabel (str): Label for the y-axis (e.g., r'$V_m$ in V').
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    model_colors = ["#CFCF00", "#1ec06f", "#ad55ff"] 
    color_idx = 0
    
    # Plot active models first
    for model_name, (t, y_vals) in data_dict.items():
        if model_name == dummy_key:
            continue # Skip the dummy for now
            
        display_name = model_name.value if hasattr(model_name, 'value') else str(model_name)
        
        # Using alpha=0.6 so the dashed dummy line can be seen if they perfectly overlap
        ax.plot(t, y_vals, label=display_name, color=model_colors[color_idx % len(model_colors)], 
                linestyle='-', linewidth=2.5, alpha=0.6)
        color_idx += 1
        
    # Plot the Dummy Reference last so it sits on top (zorder)
    if dummy_key in data_dict:
        t_dummy, y_dummy = data_dict[dummy_key]
        display_dummy = dummy_key.value if hasattr(dummy_key, 'value') else str(dummy_key)
        
        # Orange dashed line for the dummy reference
        ax.plot(t_dummy, y_dummy, label="Inactive piezo\nreference", color="#696969", 
                linestyle='--', linewidth=1.5, zorder=5)
        
    # Aesthetics
    ax.set_xlabel('Time in s', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Full bounding box and grid
    # ax.grid(True, linestyle='-', color='lightgray', alpha=0.7)
    # for spine in ax.spines.values():
    #     spine.set_edgecolor('gray')
        
    # Place legend cleanly on the right side
    ax.legend(loc='center right', fontsize=11, frameon=False)
    
    plt.tight_layout()
    # plt.savefig(filepath, dpi=300)
    plt.savefig(filepath, format="svg", bbox_inches="tight", transparent=True)
    plt.close()


def plot_single_model_time_series_comparison(data_dict, model_key, dummy_key, filepath, ylabel, mech=None, t_mech=None):
    """
    Plots a single model's time series against the dummy reference baseline.
    
    Parameters:
    data_dict (dict): Dictionary where keys are model names (or Enums) and 
                      values are tuples of (time_array, y_array).
    model_key (str/Enum): The specific model key from data_dict to plot.
    dummy_key (str/Enum): The exact key in data_dict corresponding to the dummy model.
    filepath (str): Path to save the figure.
    ylabel (str): Label for the y-axis (e.g., r'$V_m$ in V').
    mech (np.ndarray, optional): Array representing the mechanical stimuli.
    t_mech (np.ndarray, optional): Separate time vector corresponding to the mech array.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Specific color for the active model line
    model_color = PRIMARY_COLOR # Make sure PRIMARY_COLOR is defined in your environment
    
    # Plot the target model if it exists in the dictionary
    if model_key in data_dict:
        t, y_vals = data_dict[model_key]
        display_name = model_key.value if hasattr(model_key, 'value') else str(model_key)
        
        ax.plot(t, y_vals, label=display_name, color=model_color, 
                linestyle='-', linewidth=1.5, zorder=5)
    else:
        raise KeyError(f"Model key '{model_key}' not found in data_dict.")
        
    # Plot the Dummy Reference on top
    if dummy_key in data_dict:
        t_dummy, y_dummy = data_dict[dummy_key]
        
        ax.plot(t_dummy, y_dummy, label="Inactive piezo\nreference", color="#696969", 
                linestyle='--', linewidth=1.5, zorder=10)
    else:
        raise KeyError(f"Dummy key '{dummy_key}' not found in data_dict.")
        
    # Plot Mechanical Stimuli Onsets if provided
    if mech is not None and t_mech is not None:
        mech_arr = np.asarray(mech).squeeze()
        t_m = np.asarray(t_mech).squeeze()
        
        # Robust threshold-based onset detection
        threshold = 0.05 * np.max(mech_arr) if np.max(mech_arr) > 0 else 0
        is_on = (mech_arr > threshold).astype(int)
        
        diff = np.diff(is_on)
        onset_indices = np.where(diff == 1)[0] + 1
        
        if is_on[0] == 1:
            onset_indices = np.insert(onset_indices, 0, 0)
            
        onset_times = t_m[onset_indices] # Map to the separate mechanical time vector
        
        # Plot the vertical lines
        for onset in onset_times:
            ax.axvline(x=onset, color=SECONDARY_COLOR, linestyle=':', linewidth=1.5, alpha=0.7, zorder=1)
            
        # Add a single invisible line to the legend for the onsets
        if len(onset_times) > 0:
            ax.plot([], [], color=SECONDARY_COLOR, linestyle=':', linewidth=1.5, alpha=0.7, label='Mech. stim. onset')

    # Aesthetics
    ax.set_xlabel('Time in s', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
        
    # Place legend inside on the right side, WITH an opaque frame for readability
    ax.legend(loc='center right', fontsize=11, frameon=True, framealpha=1.0)
    
    plt.tight_layout()
    # plt.savefig(filepath, dpi=300)
    plt.savefig(filepath, format="svg", bbox_inches="tight", transparent=True)
    plt.close()
