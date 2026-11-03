import numpy as np
import pandas as pd

from astrocyte.piezo1_channel import DoubleGateModel, MC4StateModel, Piezo1DummyModel, SimplePiezo1Model
from analysis.time_constants import compute_activation_rise_time, compute_deactivation_tau, compute_inactivation_tau, print_statistical_summaries, print_sensitivity_analysis
from utils.plotting import plot_dataframe, plot_dataframe_comparison
from utils.plotting_expert import plot_single_model_time_series_comparison, plot_piezo_current_peaks, plot_simple_time_series, plot_spike_details, plot_tau_comparison, plot_time_series_comparison
from utils.simulation_protocol import P1ModelIdentifier, create_piezo1_simulation_protocol, samples_total, sampling_factor
from utils.timer import timer_decorator

from astrocyte.compartment import Compartment

@timer_decorator
def run_simulation(piezo_model_identifier: P1ModelIdentifier = P1ModelIdentifier.DUMMY_MODEL):
    path = "parameters.ini"

    # Create simulation protocol
    spike_train, piezo_signal = create_piezo1_simulation_protocol(piezo_model_identifier).values()

    # Init Piezo model
    match(piezo_model_identifier):

        case P1ModelIdentifier.MODEL_1:
            output_folder = "output/model_1/"
            one_sample_in_time = 1 / sampling_factor

            t_on_off = [
                (44, 44.25 + one_sample_in_time),
                (69, 69.25 + one_sample_in_time),
                (94, 94.25 + one_sample_in_time),
                (144, 144.25 + one_sample_in_time),
                (169, 169.25 + one_sample_in_time),
                (194, 194.25 + one_sample_in_time),
                (244, 244.25 + one_sample_in_time),
                (269, 269.25 + one_sample_in_time),
                (294, 294.25 + one_sample_in_time)
            ]

            piezo_model = SimplePiezo1Model(t_on_off)

        case P1ModelIdentifier.MODEL_2:
            output_folder = "output/model_2/"
            piezo_model = DoubleGateModel()

        case P1ModelIdentifier.MODEL_3:
            output_folder = "output/model_3/"
            # piezo_model = MC4StateModel(vm_0=-85e-3, p_0=0, pr_NaK=0.1, scaling=0.18)
            # piezo_model = MC4StateModel(vm_0=-85e-3, p_0=0, pr_NaK=0.1, scaling=1/(52 * 10**6))
            piezo_model = MC4StateModel(vm_0=-85e-3, p_0=0, pr_NaK=0.1)

        case P1ModelIdentifier.DUMMY_MODEL:
            output_folder = "output/model_dummy/"
            piezo_model = Piezo1DummyModel()

        case _:
            raise NotImplementedError(f"Unknown Piezo1 model identifier: {piezo_model_identifier}")

    # Init compartment
    compartment = Compartment(path, piezo_model)
    compartment.validate() #NOTE: produces an error due to inconsistent calcium
    #NOTE: corrected in the ini file

    # Solve
    solution = compartment.solve(0, samples_total/sampling_factor, spike_train, piezo_signal)
    print(f"Ratio ER: {compartment.get_ratio_er()}")

    # Create output Dataframe
    conc = np.concatenate(([solution.t], solution.y), axis=0).transpose()
    if piezo_model_identifier == P1ModelIdentifier.MODEL_2:
        cols = ["t", "Ca_i", "Ca_ER", "h", "IP3", "Na_i", "K_i", "V_m", "xn", "ma"]
    else:
        # Model 1, Model 3 and Dummy Model have no additional state variables for the Piezo channel
        cols = ["t", "Ca_i", "Ca_ER", "h", "IP3", "Na_i", "K_i", "V_m"]

    df = pd.DataFrame(conc, columns = cols)
    df["spike_train"] = spike_train
    df["mechanical_stimuli"] = piezo_signal    # add plot of mechanical stimulus
    df.to_csv(f"{output_folder}result.csv")

    # Store all tracked currents
    if piezo_model_identifier == P1ModelIdentifier.MODEL_3:
        df_tracked_currents = pd.DataFrame(
            [(k, *v.values()) for k,v in compartment.tracked_currents.items()], 
            columns=["t", "I_ncx", "I_gluT", "I_nka", "I_naleak", "I_kleak", "I_pz", "I_pz_Ca", "I_pz_Na", "I_pz_K"]
        )
    else:
        # Model 1, 2 and Dummy Model only track total Piezo current, not the individual ion currents
        df_tracked_currents = pd.DataFrame(
            [(k, *v.values()) for k,v in compartment.tracked_currents.items()], 
            columns=["t", "I_ncx", "I_gluT", "I_nka", "I_naleak", "I_kleak", "I_pz"]
        )
    
    df_tracked_currents.to_csv(f"{output_folder}all_currents_vm.csv")

    print("   Simulation completed.")


model_output_paths = {
    P1ModelIdentifier.MODEL_1: "output/model_1",
    P1ModelIdentifier.MODEL_2: "output/model_2",
    P1ModelIdentifier.MODEL_3: "output/model_3",
    P1ModelIdentifier.DUMMY_MODEL: "output/model_dummy"
}

model_ipz_labels = {
    P1ModelIdentifier.MODEL_1: r"$I_{pz}$ in A",
    P1ModelIdentifier.MODEL_2: "$I_{pz}$ in A/F",
    P1ModelIdentifier.MODEL_3: r"$I_{pz}$ in A/F",
    P1ModelIdentifier.DUMMY_MODEL: r"$I_{pz}$ in A"
}

model_mech_labels = {
    P1ModelIdentifier.MODEL_1: r"Mechanical strech $d$ in m",
    P1ModelIdentifier.MODEL_2: "Normalized active tension (NAT)",
    P1ModelIdentifier.MODEL_3: "Mechanical pressure $p$ in mmHg",
    P1ModelIdentifier.DUMMY_MODEL: "Mechanical stimulus in %"
}

model_ipz_limits = {
    P1ModelIdentifier.MODEL_1: (-1.5e-9, 1.5e-9),
    P1ModelIdentifier.MODEL_2: (-0.004, 0.004),
    P1ModelIdentifier.MODEL_3: (-4e-9, 4e-9),
    P1ModelIdentifier.DUMMY_MODEL: (-1, 1)
}

model_mech_limits = {
    P1ModelIdentifier.MODEL_1: (-7e-6, 7e-6),
    P1ModelIdentifier.MODEL_2: (-0.85, 0.85),
    P1ModelIdentifier.MODEL_3: (-85, 85),
    P1ModelIdentifier.DUMMY_MODEL: (-100, 100)
}

def read_model_data(piezo_model_identifier: P1ModelIdentifier):
    df = pd.read_csv(f"{model_output_paths[piezo_model_identifier]}/result.csv", index_col=0)
    df_tracked_currents = pd.read_csv(f"{model_output_paths[piezo_model_identifier]}/all_currents_vm.csv", index_col=0)

    return df, df_tracked_currents

def plot_all_currents(piezo_model_identifier: P1ModelIdentifier, df_tracked_currents, df_tracked_currents_reference=None, model2_transient_filter=True):
    df_tracked_currents_tmp = df_tracked_currents.copy()

    if piezo_model_identifier == P1ModelIdentifier.MODEL_3:
        naming_dict = {
            "I_ncx": {"title": r"Na/Ca Exchanger Current $I_{NCX}$", "xlabel": "Time in s", "ylabel": r"$I_{NCX}$ in A"}, 
            "I_gluT": {"title": r"GluT-dependent Pathway Current $I_{gluT}$", "xlabel": "Time in s", "ylabel": r"$I_{gluT}$ in A"},
            "I_nka": {"title": r"Na/K-ATPase Current $I_{NKA}$", "xlabel": "Time in s", "ylabel": r"$I_{NKA}$ in A"}, 
            "I_naleak": {"title": r"Na Leak Current $I_{Naleak}$", "xlabel": "Time in s", "ylabel": r"$I_{Naleak}$ in A"}, 
            "I_kleak": {"title": r"K Leak Current $I_{Kleak}$", "xlabel": "Time in s", "ylabel": r"$I_{Kleak}$ in A"}, 
            "I_pz_Ca": {"title": r"Piezo-induced Calcium Current $I_{pz,Ca}$", "xlabel": "Time in s", "ylabel": r"$I_{pz,Ca}$ in A/F"},
            "I_pz_Na": {"title": r"Piezo-induced Sodium Current $I_{pz,Na}$", "xlabel": "Time in s", "ylabel":  r"$I_{pz,Na}$ in A/F"},
            "I_pz_K": {"title": r"Piezo-induced Potassium Current $I_{pz,K}$", "xlabel": "Time in s", "ylabel":  r"$I_{pz,K}$ in A/F"}    
        }
    elif piezo_model_identifier == P1ModelIdentifier.MODEL_2:
        if model2_transient_filter:
            df_tracked_currents_tmp = df_tracked_currents_tmp[df_tracked_currents_tmp["t"] > 15]
        naming_dict = {
            "I_ncx": {"title": r"Na/Ca Exchanger Current $I_{NCX}$", "xlabel": "Time in s", "ylabel": r"$I_{NCX}$ in A"}, 
            "I_gluT": {"title": r"GluT-dependent Pathway Current $I_{gluT}$", "xlabel": "Time in s", "ylabel": r"$I_{gluT}$ in A"},
            "I_nka": {"title": r"Na/K-ATPase Current $I_{NKA}$", "xlabel": "Time in s", "ylabel": r"$I_{NKA}$ in A"}, 
            "I_naleak": {"title": r"Na Leak Current $I_{Naleak}$", "xlabel": "Time in s", "ylabel": r"$I_{Naleak}$ in A"}, 
            "I_kleak": {"title": r"K Leak Current $I_{Kleak}$", "xlabel": "Time in s", "ylabel": r"$I_{Kleak}$ in A"}, 
            "I_pz": {"title": r"Piezo Current $I_{pz}$", "xlabel": "Time in s", "ylabel": r"$I_{pz}$ in A/F"}
        }
    else:
        naming_dict = {
            "I_ncx": {"title": r"Na/Ca Exchanger Current $I_{NCX}$", "xlabel": "Time in s", "ylabel": r"$I_{NCX}$ in A"}, 
            "I_gluT": {"title": r"GluT-dependent Pathway Current $I_{gluT}$", "xlabel": "Time in s", "ylabel": r"$I_{gluT}$ in A"},
            "I_nka": {"title": r"Na/K-ATPase Current $I_{NKA}$", "xlabel": "Time in s", "ylabel": r"$I_{NKA}$ in A"}, 
            "I_naleak": {"title": r"Na Leak Current $I_{Naleak}$", "xlabel": "Time in s", "ylabel": r"$I_{Naleak}$ in A"}, 
            "I_kleak": {"title": r"K Leak Current $I_{Kleak}$", "xlabel": "Time in s", "ylabel": r"$I_{Kleak}$ in A"}, 
            "I_pz": {"title": r"Piezo Current $I_{pz}$", "xlabel": "Time in s", "ylabel": r"$I_{pz}$ in A"}
        }
    scaling_dict = None

    if df_tracked_currents_reference is None:
        plot_dataframe(df_tracked_currents_tmp, scaling_dict, naming_dict, save=True, path=f"{model_output_paths[piezo_model_identifier]}/all_currents.png", show=False)
    else:
        plot_dataframe_comparison(df_tracked_currents_tmp, df_tracked_currents_reference, scaling_dict, naming_dict, save=True, path=f"{model_output_paths[piezo_model_identifier]}/all_currents.png", show=False)
        plot_dataframe_comparison(df_tracked_currents_tmp, df_tracked_currents_reference, scaling_dict, naming_dict, save=True, path=f"{model_output_paths[piezo_model_identifier]}/all_currents.svg", show=False)
    
    print("   Created all currents plot.")
    

def create_plots(piezo_model_identifier: P1ModelIdentifier, df, df_tracked_currents):
    plot_piezo_current_peaks(
        df_tracked_currents["t"], 
        df_tracked_currents["I_pz"], 
        df["t"], 
        df["mechanical_stimuli"], 
        df["spike_train"], 
        model_ipz_labels[piezo_model_identifier],
        model_ipz_limits[piezo_model_identifier], 
        model_mech_labels[piezo_model_identifier], 
        model_mech_limits[piezo_model_identifier],
        model_output_paths[piezo_model_identifier],
        plot_title = piezo_model_identifier.value
    )
    print("   Created piezo current peaks plot.")
    
    for i in range(0, 3):
        plot_spike_details(
            df_tracked_currents["t"], 
            df_tracked_currents["I_pz"], 
            df["t"], 
            df["mechanical_stimuli"],
            mech_label = model_mech_labels[piezo_model_identifier],
            mech_unit = "" if piezo_model_identifier == P1ModelIdentifier.MODEL_2 else model_mech_labels[piezo_model_identifier].split()[-1],
            filepath = f"{model_output_paths[piezo_model_identifier]}/spikes/spike_details_glu_{i}.svg",
            plot_title = piezo_model_identifier.value,
            group_idx = i
        )
    print("   Created spike details plots for all glutamate states.")

    if piezo_model_identifier == P1ModelIdentifier.MODEL_2:
        # Plot gate time courses:
        plot_simple_time_series(
            df["t"], 
            df["ma"], 
            r"$m_a$",
            f"{model_output_paths[piezo_model_identifier]}/gate_time_course_ma.png",
            log_y = True
        )

        plot_simple_time_series(
            df["t"], 
            df["xn"], 
            r"$Xn$",
            f"{model_output_paths[piezo_model_identifier]}/gate_time_course_xn.png",
            log_y = True
        )

        # plot_dual_time_series(
        #     df["t"], 
        #     df["ma"], 
        #     df["mechanical_stimuli"],
        #     log_x = True,
        #     ylabel1 = r"$m_a$",
        #     ylabel2 = model_mech_labels[piezo_model_identifier],
        #     label1 = r"Activation Gate $m_a$",
        #     label2 = "NAT",
        #     filepath = f"{model_output_paths[piezo_model_identifier]}/gate_time_course_ma.png"
        # )

        # plot_dual_time_series(
        #     df["t"], 
        #     df["xn"], 
        #     df["V_m"],
        #     ylabel1 = r"$Xn$",
        #     ylabel2 = r"$V_m$ in V",
        #     label1 = r"Inactivation Gate $Xn$",
        #     label2 = "Membrane Potential $V_m$",
        #     filepath = f"{model_output_paths[piezo_model_identifier]}/gate_time_course_xn.png"
        # )

        print("   Created gate time course plots for Model 2.")


def main(rerun):
    # Run dummy model simulation to generate references for V_m and Ca_i
    if rerun:
        print("Running dummy model simulation...")
        run_simulation(P1ModelIdentifier.DUMMY_MODEL)

    df_dummy, df_tracked_currents_dummy = read_model_data(P1ModelIdentifier.DUMMY_MODEL)

    print("Creating plots for dummy model...")
    plot_all_currents(P1ModelIdentifier.DUMMY_MODEL, df_tracked_currents_dummy)

    vm_data = {
        P1ModelIdentifier.DUMMY_MODEL: (df_dummy["t"], df_dummy["V_m"]),
    }

    cai_data = {
        P1ModelIdentifier.DUMMY_MODEL: (df_dummy["t"], df_dummy["Ca_i"]),
    }

    nai_data = {
        P1ModelIdentifier.DUMMY_MODEL: (df_dummy["t"], df_dummy["Na_i"]),
    }

    ki_data = {
        P1ModelIdentifier.DUMMY_MODEL: (df_dummy["t"], df_dummy["K_i"]),
    }

    # Init tau data for inactivation analysis
    tau_data = {
        "activation": {
            P1ModelIdentifier.MODEL_1: {},
            P1ModelIdentifier.MODEL_2: {},
            P1ModelIdentifier.MODEL_3: {}
        },
        "inactivation": {
            P1ModelIdentifier.MODEL_1: {},
            P1ModelIdentifier.MODEL_2: {},
            P1ModelIdentifier.MODEL_3: {}
        },
        "deactivation": {
            P1ModelIdentifier.MODEL_1: {},
            P1ModelIdentifier.MODEL_2: {},
            P1ModelIdentifier.MODEL_3: {}
        }
    }
    
    glut_labels = ['[Glu] = 0.001 M', '[Glu] = 0.006', '[Glu] = 0 M']

    for model in model_output_paths.keys():
        # Loop through every model, except the dummy model
        if model == P1ModelIdentifier.DUMMY_MODEL:
            continue

        if rerun:
            print(f"Running simulation for {model.value}...")
            run_simulation(model)

        print(f"Creating plots for {model.value}...")
        df, df_tracked_currents = read_model_data(model)

        # Record Ipz effects on V_m and Ca_i
        vm_data[model] = (df["t"], df["V_m"])
        cai_data[model] = (df["t"], df["Ca_i"])

        create_plots(model, df, df_tracked_currents)
        plot_all_currents(model, df_tracked_currents, df_tracked_currents_dummy, model2_transient_filter=False)
        plot_single_model_time_series_comparison(
            data_dict=cai_data, 
            model_key=model, 
            dummy_key=P1ModelIdentifier.DUMMY_MODEL,
            filepath = f"{model_output_paths[model]}/cai_result.svg",
            ylabel=r'$[Ca^{2+}]_i$ in mM',
            mech=df["mechanical_stimuli"], 
            t_mech=df["t"]
        )

        if model == P1ModelIdentifier.MODEL_3:
            # Plot nai, and ki concentrations
            nai_data[model] = (df["t"], df["Na_i"])
            plot_single_model_time_series_comparison(
                data_dict=nai_data, 
                model_key=model, 
                dummy_key=P1ModelIdentifier.DUMMY_MODEL,
                filepath = f"{model_output_paths[model]}/nai_result.svg",
                ylabel=r'$[Na^{+}]_i$ in mM',
                mech=df["mechanical_stimuli"], 
                t_mech=df["t"]
            )

            ki_data[model] = (df["t"], df["K_i"])
            plot_single_model_time_series_comparison(
                data_dict=ki_data, 
                model_key=model, 
                dummy_key=P1ModelIdentifier.DUMMY_MODEL,
                filepath = f"{model_output_paths[model]}/ki_result.svg",
                ylabel=r'$[K^{+}]_i$ in mM',
                mech=df["mechanical_stimuli"], 
                t_mech=df["t"]
            )

        for group_idx, glut_label in enumerate(glut_labels):
            ## Get the tau data for this specific model and glutamate state
            # Activation
            print(f"Compute activation rise-time for model {model} and {glut_label}:")
            intensities_activation, rise_time_activation = compute_activation_rise_time(
                df_tracked_currents["t"], 
                df_tracked_currents["I_pz"], 
                df["t"], 
                df["mechanical_stimuli"], 
                group_idx=group_idx
            )

            # Inactivation
            print(f"Compute inactivation tau for model {model} and {glut_label}:")
            intensities_inactivation, taus_inactivation = compute_inactivation_tau(
                df_tracked_currents["t"], 
                df_tracked_currents["I_pz"], 
                df["t"], 
                df["mechanical_stimuli"], 
                group_idx=group_idx
            )

            # Deactivation
            print(f"Compute deactivation tau for model {model} and {glut_label}:")
            intensities_deactivation, taus_deactivation = compute_deactivation_tau(
                df_tracked_currents["t"], 
                df_tracked_currents["I_pz"], 
                df["t"], 
                df["mechanical_stimuli"], 
                group_idx=group_idx
            )
            
            # Save it into the nested dictionary
            tau_data["activation"][model][glut_label] = (intensities_activation, rise_time_activation)
            tau_data["inactivation"][model][glut_label] = (intensities_inactivation, taus_inactivation)
            tau_data["deactivation"][model][glut_label] = (intensities_deactivation, taus_deactivation)

    # Generate the final tau/rise-time plots
    print("Generating activation rise-time comparison...")
    output = "output/activation_rise_time.svg"
    plot_tau_comparison(tau_data["activation"], filepath=output, use_log_y=False, ylabel=r'Activation rise-time in s')
    print(f"   Activation rise-time comparison plot saved in {output}.")

    print("Generating inactivation tau comparison...")
    output = "output/inactivation_tau.svg"
    plot_tau_comparison(tau_data["inactivation"], filepath=output, use_log_y=False, ylabel=r'Inactivation time constant $\tau_{inact}$ in s')
    print(f"   Inactivation tau comparison plot saved in {output}.")

    print("Generating deactivation tau comparison...")
    output = "output/deactivation_tau.svg"
    tau_data["deactivation"][P1ModelIdentifier.MODEL_1] = {}
    plot_tau_comparison(tau_data["deactivation"], filepath=output, use_log_y=False, ylabel=r'Deactivation time constant $\tau_{deact}$ in s', start_0=True)
    print(f"   Deactivation tau comparison plot saved in {output}.")

    # Time constant analyses
    print_statistical_summaries(tau_data)
    print_sensitivity_analysis(tau_data)

    # Generate effects plots for V_m and Ca_i
    print("Generating effects plots for V_m and Ca_i...")

    output = "output/vm_comparison.svg"
    plot_time_series_comparison(
        data_dict = vm_data, 
        dummy_key = P1ModelIdentifier.DUMMY_MODEL, 
        filepath = output,
        ylabel=r'$V_m$ in V'
    )
    print(f"   V_m comparison plot saved in {output}.")

    output = "output/cai_comparison.svg"
    plot_time_series_comparison(
        data_dict = cai_data, 
        dummy_key = P1ModelIdentifier.DUMMY_MODEL, 
        filepath = output,
        ylabel=r'$[Ca^{2+}]_i$ in mM'
    )
    print(f"   Ca_i comparison plot saved in {output}.")

if __name__ == "__main__":
    main(rerun=False)    
