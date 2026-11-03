import numpy as np

def _get_stimulus_group(t2, mech, group_idx=0):
    """
    Helper function to detect mechanical stimuli and isolate a specific group of spikes.
    
    Args:
        t2 (array): Time array for the mechanical stimulus.
        mech (array): Mechanical stimulus amplitude array.
        group_idx (int): Which third of the stimuli to extract (0, 1, or 2).
        
    Returns:
        group_onsets (list): Indices in t2/mech where the stimuli turn on.
        group_offsets (list): Indices in t2/mech where the stimuli turn off.
        intensities (list): Maximum mechanical amplitude for each stimulus in the group.
    """
    # Ensure inputs are 1D arrays to prevent indexing errors
    mech = np.asarray(mech).squeeze()
    t2 = np.asarray(t2).squeeze()
    
    # Define a dynamic threshold to identify when the stimulus is "on"
    threshold = 0.05 * np.max(mech) if np.max(mech) > 0 else 0
    is_on = (mech > threshold).astype(int)
    
    # Detect edges: +1 indicates stimulus onset, -1 indicates stimulus offset
    diff = np.diff(is_on)
    onsets = np.where(diff == 1)[0] + 1
    offsets = np.where(diff == -1)[0] + 1
    
    # Handle edge cases where the simulation starts or ends while the stimulus is already active
    if is_on[0] == 1: onsets = np.insert(onsets, 0, 0)
    if is_on[-1] == 1: offsets = np.append(offsets, len(mech) - 1)
        
    num_stimuli = len(onsets)
    if num_stimuli == 0: 
        return [], [], []
        
    # Divide the detected stimuli into 3 groups and select the requested one
    stims_per_group = max(1, num_stimuli // 3)
    start_idx = group_idx * stims_per_group
    end_idx = min((group_idx + 1) * stims_per_group, num_stimuli)
    
    group_onsets = onsets[start_idx:end_idx]
    group_offsets = offsets[start_idx:end_idx]
    
    # Extract the peak pressure/stretch applied during each active window
    intensities = [np.max(mech[on:off]) for on, off in zip(group_onsets, group_offsets)]
    
    return group_onsets, group_offsets, intensities


def compute_activation_rise_time(t1, ipz, t2, mech, group_idx=0):
    """
    Computes the macroscopic 10-90% rise time during the activation phase.
    This bypasses exponential curve fitting, making it robust against sigmoidal onsets.
    Relevant timeframe: t_onset until t_max_downward_peak
    """
    t1 = np.asarray(t1).squeeze()
    ipz = np.asarray(ipz).squeeze()

    group_onsets, group_offsets, intensities = _get_stimulus_group(t2, mech, group_idx)
    rise_times = []
    
    for onset_idx, offset_idx in zip(group_onsets, group_offsets):
        t_onset, t_offset = t2[onset_idx], t2[offset_idx]
        intensity = np.max(mech[onset_idx:offset_idx])
        
        # Isolate the exact window where the mechanical stimulus is applied
        mask_t1 = (t1 >= t_onset) & (t1 <= t_offset) 
        if not np.any(mask_t1):
            print(f"   [Activation] Failed for {intensity:g}: No data in stimulus window.")
            rise_times.append(np.nan)
            continue
            
        t_snip, ipz_snip = t1[mask_t1], ipz[mask_t1]
        
        # Calculate resting baseline just before the stimulus and shift the signal to 0.0
        pre_stim_mask = (t1 >= t_onset - 0.5) & (t1 < t_onset)
        if np.any(pre_stim_mask):
            baseline = np.mean(ipz[pre_stim_mask])
        else:
            valid_idx = np.where(t1 <= t_onset)[0]
            baseline = ipz[valid_idx[-1]] if len(valid_idx) > 0 else ipz[0]
            
        ipz_corr = ipz_snip - baseline
        
        # Locate the maximum absolute current (works for both inward and outward currents)
        peak_idx = np.argmax(np.abs(ipz_corr))
        peak_val = ipz_corr[peak_idx]
        
        # Restrict analysis strictly to the activation phase (onset to peak)
        t_act, ipz_act = t_snip[:peak_idx+1], ipz_corr[:peak_idx+1]
        
        # Define 10% and 90% amplitude thresholds
        target_10, target_90 = peak_val * 0.10, peak_val * 0.90
        
        def find_crossing(t_arr, y_arr, target, is_positive):
            """Finds the precise sub-timestep where the signal crosses a target amplitude."""
            crossings = np.where(y_arr >= target)[0] if is_positive else np.where(y_arr <= target)[0]
            if len(crossings) > 0:
                idx = crossings[0]
                if idx == 0: return t_arr[0]
                
                # Linear interpolation for sub-millisecond precision
                t_c1, t_c2, y_c1, y_c2 = t_arr[idx-1], t_arr[idx], y_arr[idx-1], y_arr[idx]
                if y_c1 == y_c2: return t_c1
                return t_c1 + (target - y_c1) * (t_c2 - t_c1) / (y_c2 - y_c1)
            return np.nan

        is_pos = peak_val > 0
        t_10 = find_crossing(t_act, ipz_act, target_10, is_pos)
        t_90 = find_crossing(t_act, ipz_act, target_90, is_pos)
        
        if not np.isnan(t_10) and not np.isnan(t_90):
            rise_times.append(t_90 - t_10)
        else:
            print(f"   [Activation] Failed for {intensity:g}: Signal did not reach 10% or 90% threshold.")
            rise_times.append(np.nan)
            
    return intensities, rise_times


def compute_inactivation_tau(t1, ipz, t2, mech, group_idx=0):
    """
    Computes the decay time constant (tau, 63.2% decay) for the macroscopic 
    inactivation phase while the mechanical stimulus is still active.
    Relevant timeframe: t_max_downward_peak until t_offset
    """
    t1 = np.asarray(t1).squeeze()
    ipz = np.asarray(ipz).squeeze()

    group_onsets, group_offsets, intensities = _get_stimulus_group(t2, mech, group_idx)
    taus = []
    
    for onset_idx, offset_idx in zip(group_onsets, group_offsets):
        t_onset, t_offset = t2[onset_idx], t2[offset_idx]
        intensity = np.max(mech[onset_idx:offset_idx])
        
        # Isolate the stimulus window
        mask_t1 = (t1 >= t_onset) & (t1 <= t_offset) 
        if not np.any(mask_t1):
            print(f"   [Inactivation] Failed for {intensity:g}: No data in stimulus window.")
            taus.append(np.nan)
            continue
            
        t_snip, ipz_snip = t1[mask_t1], ipz[mask_t1]
        
        # Shift current trace so the pre-stimulus baseline sits at exactly 0.0
        pre_stim_mask = (t1 >= t_onset - 0.5) & (t1 < t_onset)
        if np.any(pre_stim_mask):
            baseline = np.mean(ipz[pre_stim_mask])
        else:
            valid_idx = np.where(t1 <= t_onset)[0]
            baseline = ipz[valid_idx[-1]] if len(valid_idx) > 0 else ipz[0]
            
        ipz_corr = ipz_snip - baseline
        
        peak_idx = np.argmax(np.abs(ipz_corr))
        peak_val, t_peak = ipz_corr[peak_idx], t_snip[peak_idx]
        
        # Restrict analysis strictly to the inactivation phase (peak to offset)
        t_inact, ipz_inact = t_snip[peak_idx:], ipz_corr[peak_idx:]
        
        # If the peak occurred at the very end of the stimulus, no inactivation happened
        if len(t_inact) <= 1:
            print(f"   [Inactivation] Failed for {intensity:g}: Peak at stimulus offset (no inactivation phase).")
            taus.append(np.nan)
            continue
            
        # Standard tau: signal decays by 1 - (1/e) ≈ 63.2%, leaving 36.8% of the peak
        target_val = peak_val * 0.368
        crossings = np.where(ipz_inact <= target_val)[0] if peak_val > 0 else np.where(ipz_inact >= target_val)[0]
            
        if len(crossings) > 0:
            idx = crossings[0]
            if idx == 0:
                print(f"   [Inactivation] Noted for {intensity:g}: Instantaneous decay (tau = 0.0).")
                taus.append(0.0)
            else:
                # Sub-timestep interpolation for precision
                t_c1, t_c2, y_c1, y_c2 = t_inact[idx-1], t_inact[idx], ipz_inact[idx-1], ipz_inact[idx]
                t_exact = t_c1 if y_c1 == y_c2 else t_c1 + (target_val - y_c1) * (t_c2 - t_c1) / (y_c2 - y_c1)
                
                # Tau is relative to the peak time, not stimulus onset
                taus.append(t_exact - t_peak)
        else:
            print(f"   [Inactivation] Failed for {intensity:g}: Did not reach 63.2% decay before stimulus ended.")
            taus.append(np.nan) # Inactivation did not reach 63.2% before stimulus ended
            
    return intensities, taus


def compute_deactivation_tau(t1, ipz, t2, mech, group_idx=0):
    """
    Computes the decay time constant (tau, 63.2% decay) for the closing relaxation 
    (deactivation) phase entirely AFTER the mechanical stimulus is removed.
    Relevant timeframe: t_offset until t_window_end
    """
    t1 = np.asarray(t1).squeeze()
    ipz = np.asarray(ipz).squeeze()

    group_onsets, group_offsets, intensities = _get_stimulus_group(t2, mech, group_idx)
    taus = []
    
    for onset_idx, offset_idx in zip(group_onsets, group_offsets):
        t_onset, t_offset = t2[onset_idx], t2[offset_idx]
        intensity = np.max(mech[onset_idx:offset_idx])
        
        # Extract a wide window extending well past the stimulus to capture the full tail
        mask_t1 = (t1 >= t_onset - 0.5) & (t1 <= t_offset + 5.0) 
        if not np.any(mask_t1):
            print(f"   [Deactivation] Failed for {intensity:g}: No data in window.")
            taus.append(np.nan)
            continue
            
        t_snip, ipz_snip = t1[mask_t1], ipz[mask_t1]
        
        # Shift trace so baseline is 0.0
        pre_stim_mask = (t_snip >= t_onset - 0.5) & (t_snip < t_onset)
        if np.any(pre_stim_mask):
            baseline = np.mean(ipz_snip[pre_stim_mask])
        else:
            valid_idx = np.where(t_snip <= t_onset)[0]
            baseline = ipz_snip[valid_idx[-1]] if len(valid_idx) > 0 else ipz_snip[0]
            
        ipz_corr = ipz_snip - baseline
        
        # Find the exact index where the stimulus officially turns off
        offset_mask = t_snip >= t_offset
        if not np.any(offset_mask):
            print(f"   [Deactivation] Failed for {intensity:g}: No data found after stimulus offset.")
            taus.append(np.nan)
            continue
            
        deact_start_idx = np.argmax(offset_mask) 
        
        # The deactivation "peak" is simply whatever the current is at t_offset
        start_val, t_start = ipz_corr[deact_start_idx], t_snip[deact_start_idx]
        target_val = start_val * 0.368
        
        # Isolate strictly the data after the stimulus is gone
        t_after, ipz_after = t_snip[deact_start_idx:], ipz_corr[deact_start_idx:]
        
        crossings = np.where(ipz_after <= target_val)[0] if start_val > 0 else np.where(ipz_after >= target_val)[0]
            
        if len(crossings) > 0:
            idx = crossings[0]
            if idx == 0:
                print(f"   [Deactivation] Noted for {intensity:g}: Instantaneous deactivation (tau = 0.0). Will not appear on log scale.")
                taus.append(0.0) # Instantaneous deactivation (e.g., SimplePiezo1Model)
            else:
                # Sub-timestep interpolation for precision
                t_c1, t_c2, y_c1, y_c2 = t_after[idx-1], t_after[idx], ipz_after[idx-1], ipz_after[idx]
                t_exact = t_c1 if y_c1 == y_c2 else t_c1 + (target_val - y_c1) * (t_c2 - t_c1) / (y_c2 - y_c1)
                
                # Tau is relative to the offset time, not the absolute peak
                taus.append(t_exact - t_start)
        else:
            print(f"   [Deactivation] Failed for {intensity:g}: Did not reach 63.2% decay within 5.0s window.")
            taus.append(np.nan) # Deactivation did not reach 63.2% within the +5.0s window
            
    return intensities, taus


import numpy as np

def print_statistical_summaries(tau_data):
    """
    Parses the tau_data dictionary and prints mean & standard deviation 
    for the kinetic phases using 3-space indentation for hierarchy.
    """
    
    # Extract phases and models dynamically based on the dictionary structure
    phases = list(tau_data.keys())
    if not phases:
        return
    models = list(tau_data[phases[0]].keys())
    
    print("OVERALL SUMMARY of TAU/RISE-TIME data (Grouped by Model and Phase)")
    for model in models:
        print(f"   Model: {model}")
        for phase in phases:
            all_taus = []
            # Access data via tau_data[phase][model]
            for glut_label, (intensities, taus) in tau_data[phase][model].items():
                all_taus.extend(taus)
            
            valid_taus = [t for t in all_taus if not np.isnan(t)]
            
            if valid_taus:
                mean_val = np.nanmean(valid_taus) * 1000
                std_val = np.nanstd(valid_taus) * 1000
                print(f"      {phase.capitalize():<12} : {mean_val:6.2f} ± {std_val:.2f} ms")
            else:
                print(f"      {phase.capitalize():<12} : No valid data")

    print("\nDETAILED SUMMARY of TAU/RISE-TIME data (Grouped by Model, Phase, and [Glu])")
    for model in models:
        print(f"   Model: {model}")
        for phase in phases:
            print(f"      Phase: {phase.capitalize()}")
            for glut_label, (intensities, taus) in tau_data[phase][model].items():
                
                valid_taus = [t for t in taus if not np.isnan(t)]
                
                if valid_taus:
                    mean_val = np.nanmean(valid_taus) * 1000
                    std_val = np.nanstd(valid_taus) * 1000
                    print(f"         {glut_label:<15} : {mean_val:6.2f} ± {std_val:.2f} ms")
                else:
                    print(f"         {glut_label:<15} : No valid data")

def print_sensitivity_analysis(tau_data):
    """
    Parses the tau_data dictionary and prints the maximum relative sensitivity
    (percentage change) of the kinetic phases to Mechanical Stimulus and 
    Glutamate Concentration, using the mathematical minimum as the baseline.
    """
    
    # Extract phases and models dynamically based on the dictionary structure
    phases = list(tau_data.keys())
    if not phases:
        return
    models = list(tau_data[phases[0]].keys())
    
    print("SENSITIVITY ANALYSIS (Maximum Relative Deviation)")
    
    for model in models:
        print(f"   Model: {model}")
        for phase in phases:
            print(f"      Phase: {phase.capitalize()}")
            
            # Access data via tau_data[phase][model]
            glut_data = tau_data[phase][model]
            
            if not glut_data:
                print("         No data available")
                continue
                
            # 1. Sensitivity to Mechanical Stimulus
            # (Hold Glutamate constant, sweep Mechanical)
            mech_sensitivities = []
            for glut_label, (intensities, taus) in glut_data.items():
                valid_taus = [t for t in taus if not np.isnan(t)]
                if valid_taus:
                    t_max, t_min = np.max(valid_taus), np.min(valid_taus)
                    # Prevent division by zero if tau is perfectly 0.0
                    denom = max(t_min, 1e-15) 
                    delta = ((t_max - t_min) / denom) * 100
                    mech_sensitivities.append(delta)
                    
            max_mech_delta = np.max(mech_sensitivities) if mech_sensitivities else np.nan
            
            # 2. Sensitivity to Glutamate
            # (Hold Mechanical constant, sweep Glutamate)
            # Find the number of mechanical intensity states (e.g., 3: low, med, high)
            num_intensities = 0
            for _, (intensities, taus) in glut_data.items():
                num_intensities = len(taus)
                break
                
            glut_sensitivities = []
            for i in range(num_intensities):
                taus_at_intensity_i = []
                for glut_label, (intensities, taus) in glut_data.items():
                    if i < len(taus):
                        t = taus[i]
                        if not np.isnan(t):
                            taus_at_intensity_i.append(t)
                            
                if taus_at_intensity_i:
                    t_max, t_min = np.max(taus_at_intensity_i), np.min(taus_at_intensity_i)
                    denom = max(t_min, 1e-15)
                    delta = ((t_max - t_min) / denom) * 100
                    glut_sensitivities.append(delta)
                    
            max_glut_delta = np.max(glut_sensitivities) if glut_sensitivities else np.nan
            
            # Formatting the output strings
            if not np.isnan(max_mech_delta):
                # If the change is virtually zero, report it as < 0.01%
                mech_str = f"{max_mech_delta:6.2f} %" if max_mech_delta >= 0.01 else "< 0.01 %"
            else:
                mech_str = "N/A"
                
            if not np.isnan(max_glut_delta):
                glut_str = f"{max_glut_delta:6.2f} %" if max_glut_delta >= 0.01 else "< 0.01 %"
            else:
                glut_str = "N/A"
                
            print(f"         Max Δ due to Mech. Stimulus : {mech_str}")
            print(f"         Max Δ due to Glutamate      : {glut_str}")
