import math
from configparser import ConfigParser

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interpolate

from astrocyte.piezo1_channel import Piezo1Channel, PiezoModelStrategy
import constants
from astrocyte.endoplasmatic_reticulum import EndoplasmaticReticulum
from astrocyte.extracellular_space import ExtracellularSpace
from astrocyte.glut_pathway import GluTPathway
from astrocyte.intracellular_space import IntracellularSpace
from astrocyte.ip3_production import IP3Production
from astrocyte.mglur_pathway import MGluRPathway
from utils import config_editor
from utils.writable import Writable


class Compartment(Writable):
    # NOTE : the variables should be initialized using the .ini file
    # svr = 5e6  # Surface Volume Ratio. Used to compute ratio_er. The CLOSER to the soma, the SMALLER this value, [A / Vol] = [1/m]
    # calcium_tot = 1.82073  # tot Ca^2+ in this compartment
    # sodium_tot = 165  # tot Na^+ in this compartment
    # potassium_tot = 103  # tot K^+ in this compartment
    # membrane_capacitance = 0.01  # TODO does this belong here.
    # voltage = -85e-3  # Membrane Voltage

    def __init__(self, path, piezo_model: PiezoModelStrategy):
        self._name = "Compartment"
        self.from_file(path) # this will initialize the variables for the class

        # Init Piezo1 Channel with provided model
        self.piezo_channel = Piezo1Channel(piezo_model)

        ratio_er = self.get_ratio_er()
        sqrt_er = math.sqrt(ratio_er)
        self.er = EndoplasmaticReticulum(self.svr * sqrt_er / (constants.FARADAY_CONSTANT * ratio_er))
        self.extra = ExtracellularSpace()

        self.intra = IntracellularSpace(self.svr / (constants.FARADAY_CONSTANT * (1 - ratio_er)),
                                        self.svr * sqrt_er / (constants.FARADAY_CONSTANT * (1 - ratio_er)),
                                        self.piezo_channel.get_intracellular_ca_effect_factor(self.v_comp, self.svr/1e6, ratio_er),
                                        self.piezo_channel.get_intracellular_na_k_effect_factor(self.svr/1e6, ratio_er))

        self.gluT_pathway = GluTPathway(self.intra, self.extra)
        self.ip3_production = IP3Production(self.intra, self.extra)
        self.mglur_pathway = MGluRPathway(self.intra, self.er, (1 - ratio_er) / (self.svr * sqrt_er))

        self.interpolation_f = None

        # TODO: Add further configuration for piezo channel here if necessary (e.g., add intra-/extracellular space)
        if self.piezo_channel.add_intra_extra(self.intra, self.extra):
            print(f"Added intra-/extracellular space to Piezo1 channel")

        N_Pz1 = self.v_comp * self.svr/1e6 * self.rho_pz1
        if self.piezo_channel.set_N_Pz1(N_Pz1):
            print(f"Set N_Pz1 = {N_Pz1} in Piezo1 channel.")

        g_pz1 = self.piezo_channel.configure_total_conductance(self.v_comp, self.svr/1e6, self.rho_pz1, self.gamma_pz1, self.membrane_capacitance)
        if g_pz1 is not None:
            print(f"Set total conductance to {g_pz1}")

        self.interpolation_mechanical_stimuli = None
        
        self.from_file(path) # this will update the variables of all subcalsses

        self.tracked_currents = {}

    def get_name(self):
        return self._name
    
    def get_ratio_er(self):
        """
        Computes the volume ratio of the ER in comparison to the volume of the intracellular space. Depends on SVR (Surface Volume Ratio).
        The CLOSER to the Soma, the LARGER this value. Compartments close to the soma should have a ratio of about 0.15 (SVR 38e6)
        The FURTHER AWAY from the soma, the SMALLER (0) this value
        Reason: Astrocytic compartments close to the synapse do not contain an internal Ca^2+ store.
        :return:
            Volume Ratio ER / Intra Cellular Space.
        """
        return round(0.15 * math.exp(-0.002 * (self.svr / 1e6) ** 2.32), 4)
        # return 0.15 # represented ratio er in the soma
        # return 0.000000000000000001 # set location close to PAPs

    def solve(self, start, end, spike_train, mechanical_stimuli):
        initial_values = [self.intra.calcium, self.er.calcium, self.mglur_pathway.ip3.h, self.intra.ip3, self.intra.sodium, self.intra.potassium,
                          self.voltage]

        try:
            initial_values.extend([self.piezo_channel.get_double_gate_model().xn, self.piezo_channel.get_double_gate_model().m_a])
        except NotImplementedError:
            pass

        t = np.linspace(start, end, len(spike_train))

        # Glutamate interpolation
        self.interpolation_f = interpolate.interp1d(t, spike_train)

        # Mechanical stimulus interpolation
        self.interpolation_mechanical_stimuli = interpolate.interp1d(t, mechanical_stimuli)

        sol = solve_ivp(self.dxdt, (start, end), initial_values, method='Radau', t_eval=t, max_step=0.001)  # for some reason, higher order RK fails?
        return sol

    def dxdt(self, t, x):
        """

        :param x: x(0) internal calcium, x(1) er calcium, x(2) h, x(3) internal ip3, x(4) internal sodium, x(5) internal potassium, x(6) membrane voltage
        :param t: time steps to evaluate
        :return:

        """
        self.update(x, t)

        mglur_currents = self.mglur_pathway.compute_currents()
        gluT_currents = self.gluT_pathway.compute_currents(self.voltage)
        ip3_measures = self.ip3_production.compute_measures()
        piezo_currents = self.piezo_channel.compute_currents(self.voltage, t, self.interpolation_mechanical_stimuli([t])[0])

        try:
            # Try accumulated current
            i_pz =piezo_currents["I_pz"]

            odes = [
                self.intra.change_in_calcium_ode(gluT_currents["NCX"], mglur_currents["IP3"], mglur_currents["SERCAPump"], mglur_currents["CaLeak"], i_pz),
                self.er.change_in_calcium_ode(mglur_currents["IP3"], mglur_currents["SERCAPump"], mglur_currents["CaLeak"]),
                self.mglur_pathway.ip3.change_in_h_ode(self.intra.ip3, self.intra.calcium),
                self.intra.change_in_ip3_ode(ip3_measures["PLCBeta"], ip3_measures["PLCDelta"], ip3_measures["IP33K"], ip3_measures["IP5P"]),
                self.intra.change_in_sodium_ode(gluT_currents["GluT"], gluT_currents["NKA"], gluT_currents["NCX"], gluT_currents["SodiumLeak"]),
                self.intra.change_in_potassium_ode(gluT_currents["GluT"], gluT_currents["NKA"], gluT_currents["PotassiumLeak"]),
            ]

            try:
                # try if double gate model variables available   -->   P1.dual-vm
                double_gates = [
                    self.piezo_channel.get_double_gate_model().change_in_xn_ode(self.voltage),
                    self.piezo_channel.get_double_gate_model().change_in_ma_ode(self.interpolation_mechanical_stimuli([t])[0])
                ]

                odes.append(
                    self.change_in_voltage_ode(mglur_currents["IP3"], mglur_currents["SERCAPump"], 
                    mglur_currents["CaLeak"], gluT_currents["NCX"], gluT_currents["GluT"], gluT_currents["NKA"], 
                    gluT_currents["SodiumLeak"], gluT_currents["PotassiumLeak"], I_pz_Ca_AF=i_pz)
                )

                odes.extend(double_gates)

            except NotImplementedError:
                # P1.dual
                odes.append(
                    self.change_in_voltage_ode(mglur_currents["IP3"], mglur_currents["SERCAPump"], 
                    mglur_currents["CaLeak"], gluT_currents["NCX"], gluT_currents["GluT"], gluT_currents["NKA"], 
                    gluT_currents["SodiumLeak"], gluT_currents["PotassiumLeak"], I_pz_Ca_A=i_pz)
                )

            self.tracked_currents[t] = {"I_ncx": gluT_currents["NCX"], "I_gluT":  gluT_currents["GluT"], "I_nka": gluT_currents["NKA"],
                                        "I_naleak": gluT_currents["SodiumLeak"], "I_kleak": gluT_currents["PotassiumLeak"], 
                                        "I_pz": i_pz}

        except KeyError:
            # Split the piezo current into its components if the model does not provide accumulated current
            odes = [
                self.intra.change_in_calcium_ode(gluT_currents["NCX"], mglur_currents["IP3"], mglur_currents["SERCAPump"], mglur_currents["CaLeak"], piezo_currents["I_Ca"]),
                self.er.change_in_calcium_ode(mglur_currents["IP3"], mglur_currents["SERCAPump"], mglur_currents["CaLeak"]),
                self.mglur_pathway.ip3.change_in_h_ode(self.intra.ip3, self.intra.calcium),
                self.intra.change_in_ip3_ode(ip3_measures["PLCBeta"], ip3_measures["PLCDelta"], ip3_measures["IP33K"], ip3_measures["IP5P"]),
                self.intra.change_in_sodium_ode(gluT_currents["GluT"], gluT_currents["NKA"], gluT_currents["NCX"], gluT_currents["SodiumLeak"], piezo_currents["I_Na"]),
                self.intra.change_in_potassium_ode(gluT_currents["GluT"], gluT_currents["NKA"], gluT_currents["PotassiumLeak"], piezo_currents["I_K"]),
                self.change_in_voltage_ode(mglur_currents["IP3"], mglur_currents["SERCAPump"], mglur_currents["CaLeak"], gluT_currents["NCX"],
                                        gluT_currents["GluT"], gluT_currents["NKA"], gluT_currents["SodiumLeak"], gluT_currents["PotassiumLeak"], 
                                        I_pz_Ca_AF=piezo_currents["I_Ca"], I_pz_Na_AF=piezo_currents["I_Na"], I_pz_K_AF=piezo_currents["I_K"])
                # self.change_in_voltage_ode(mglur_currents["IP3"], mglur_currents["SERCAPump"], mglur_currents["CaLeak"], gluT_currents["NCX"],
                #                         gluT_currents["GluT"], gluT_currents["NKA"], gluT_currents["SodiumLeak"], gluT_currents["PotassiumLeak"], 
                #                         piezo_currents["I_Ca"] * 10**-6, piezo_currents["I_Na"] * 10**-6, piezo_currents["I_K"] * 10**-6)
            ]

            self.tracked_currents[t] = {"I_ncx": gluT_currents["NCX"], "I_gluT":  gluT_currents["GluT"], "I_nka": gluT_currents["NKA"],
                                        "I_naleak": gluT_currents["SodiumLeak"], "I_kleak": gluT_currents["PotassiumLeak"], 
                                        "I_pz": piezo_currents["I_Ca"] + piezo_currents["I_Na"] + piezo_currents["I_K"],
                                        "I_pz_Ca": piezo_currents["I_Ca"], "I_pz_Na": piezo_currents["I_Na"], "I_pz_K": piezo_currents["I_K"]}

        return odes

    def update(self, x, t):
        epsilon = 0.0000001
        self.extra.glu = self.interpolation_f([t])[0]
        self.intra.calcium = max(x[0], epsilon)
        self.er.calcium = max(x[1], epsilon)
        self.extra.calcium = self.calcium_tot - self.intra.calcium - self.er.calcium
        self.mglur_pathway.ip3.h = x[2]
        self.intra.ip3 = x[3]

        self.intra.sodium = min(max(x[4], epsilon), self.sodium_tot - epsilon)
        self.extra.sodium = self.sodium_tot - self.intra.sodium

        self.intra.potassium = min(max(x[5], epsilon), self.potassium_tot - epsilon)
        self.extra.potassium = self.potassium_tot - self.intra.potassium

        self.voltage = x[6]

        try:
            self.piezo_channel.get_double_gate_model().xn = x[7]
            self.piezo_channel.get_double_gate_model().m_a = x[8]
        except (IndexError, NotImplementedError):
            pass

        self.validate()

    def validate(self):
        """
        Ensure that Ca/Na/K values are consistent.
        :return:
        """
        assert math.isclose(self.sodium_tot - self.intra.sodium, self.extra.sodium), "Na (intra,extra) do not add up to tot"
        assert math.isclose(self.potassium_tot - self.intra.potassium, self.extra.potassium), "K (intra, extra) do not add up to tot"
        assert math.isclose(self.calcium_tot - self.intra.calcium - self.er.calcium, self.extra.calcium), f"CA (intra,ER,extra) do not add up to tot {self.calcium_tot,self.intra.calcium,self.er.calcium, self.extra.calcium}"
        
    def change_in_voltage_ode(self, I_ip3, I_serca, I_caleak, I_ncx, I_gluT, I_nka, I_naleak, I_kleak, I_pz_Ca_A=0, I_pz_Ca_AF=0, I_pz_Na_AF=0, I_pz_K_AF=0):
        # return - 1. / self.membrane_capacitance * (- 2 * I_ip3 + 2 * I_serca - 2 * I_caleak + I_ncx - 2 * I_gluT + I_nka + I_naleak + I_kleak + I_Ca + I_Na + I_K)
        return - 1. / self.membrane_capacitance * (I_ncx - 2 * I_gluT + I_nka + I_naleak + I_kleak + I_pz_Ca_A) - (I_pz_Ca_AF + I_pz_Na_AF + I_pz_K_AF)

    def to_file(self, path):
        config_parser = ConfigParser()
        config_editor.write_recursively(self, config_parser)
        with open(path, 'w') as f:
            config_parser.write(f)

    def from_file(self, path):
        config_parser = ConfigParser(inline_comment_prefixes="#")
        config_parser.read(path)
        config_editor.read_recursively(self, config_parser)

