from abc import ABC, abstractmethod
import math
import numpy as np
import scipy as sp

from astrocyte.extracellular_space import ExtracellularSpace
from astrocyte.intracellular_space import IntracellularSpace
import constants
import simulation_config


class PiezoModelStrategy(ABC):
    """
    Provides common functionality to all strategies of Piezo1 models.    
    """

    def __init__(self, name) -> None:
        super().__init__()
        self._name = name
        self._current: dict[float, float] = {}
    
    @abstractmethod
    def compute_currents(self, vm, t, mechanical_stimulus) -> dict[str, float]:
        pass

    def get_name(self) -> str:
        return self._name

class Piezo1Channel():
    """
    Provides access to the Piezo1 channel's functionality, i.e. ion currents.
    """

    def __init__(self, model_strategy: PiezoModelStrategy) -> None:
        self._model_strategy = model_strategy
        print(f"Initialized Piezo1 channel for model >>{self._model_strategy.get_name()}<<")

    @property
    def model_strategy(self) -> PiezoModelStrategy:
        """
        Implementation strategy of the Piezo1 model.
        """
        return self._model_strategy
    
    @model_strategy.setter
    def model_strategy(self, model_strategy: PiezoModelStrategy) -> None:
        self._model_strategy = model_strategy

    def compute_currents(self, vm, t, mechanical_stimulus):
        """
        Computes the ion currents as provided from the set Piezo1 model.
        
        Parameters:
            vm -- membrane voltage
            t  -- time of evaluation
            mechanical_stimulus -- Mechanical stimulation signal for piezo reaction

        Returns:
            dict[str, float] -- dictionary of computed ion currents, should contain at least 'I_pz' for Piezo1 current
        """
        return self._model_strategy.compute_currents(vm, t, mechanical_stimulus)

    def get_double_gate_model(self) -> 'DoubleGateModel':
        if isinstance(self.model_strategy, DoubleGateModel):
            return self.model_strategy
    
        raise NotImplementedError("DoubleGateModel not implemented in Piezo1 channel!")
    
    def add_intra_extra(self, intra: IntracellularSpace, extra: ExtracellularSpace):
        """
        Add intra- and extracellular space. Necessary for Markov Chain Model

        returns:
            bool - True, if variables intra and extra could be passed to MCM
        """
        isMCM = isinstance(self.model_strategy, MC4StateModel)
        
        if isMCM:
            self.model_strategy.intra = intra
            self.model_strategy.extra = extra

        return isMCM

    def set_N_Pz1(self, N_Pz1):
        """
        The MCM requires a fixed number of Piezo1 channels. Set it with this function.

        returns: bool - True, if N_Pz1 could be passed to MCM
        """
        isMCM = isinstance(self.model_strategy, MC4StateModel)
                
        if isMCM:
            self.model_strategy.init_conductances(N_Pz1)

        return isMCM

    def configure_total_conductance(self, V_comp, svr, rho_Pz1, gamma_Pz1_nS, membrane_capacitance_pf_per_um2):
        """
        Configures the total Piezo1 conductance dynamically based on the model volume/surface.
        Only necessary in HH-like models, since in MC models this connection to the astrocyte model
        is set via the #channel parameter.

        returns: set conductance value
        """
        isP1dual = isinstance(self.model_strategy, SimplePiezo1Model)
        isP1dual_vm = isinstance(self.model_strategy, DoubleGateModel)

        if isP1dual:
            conductance = V_comp * svr * rho_Pz1 * gamma_Pz1_nS
            self.model_strategy.set_G_pz(conductance)
            return conductance

        if isP1dual_vm:
            gamma_Pz1_pS = gamma_Pz1_nS * 1000
            conductance = (rho_Pz1 * gamma_Pz1_pS) / membrane_capacitance_pf_per_um2
            self.model_strategy.set_g_p(conductance)
            return conductance

        return None


    def get_intracellular_ca_effect_factor(self, V_comp=None, svr=None, ratio_er=None):
        """
        Returns the factor for intracellular calcium effect of piezo current
        """
        P_f = 0.15      # Ca portion of I_pz

        if isinstance(self.model_strategy, SimplePiezo1Model) and V_comp is not None and ratio_er is not None:
            factor = 5.182 * 10**12
            return -factor * P_f / (V_comp * (1 - ratio_er))
        elif isinstance(self.model_strategy, DoubleGateModel) and svr is not None and ratio_er is not None:
            # gamma_A = 1
            # return svr / (2 * gamma_A * constants.FARADAY_CONSTANT * (1 - ratio_er))
            factor = 0.05182
            return -factor * P_f * svr / (1 - ratio_er) 
        elif isinstance(self.model_strategy, Piezo1DummyModel):
            return 0
        elif isinstance(self.model_strategy, MC4StateModel) and svr is not None and ratio_er is not None:
            factor = 0.05182
            return -factor * svr / (1 - ratio_er) 
        
        raise NotImplementedError("Ca effect factor not implemented in Piezo1 channel!")

    def get_intracellular_na_k_effect_factor(self, svr=None, ratio_er=None):
        """
        Returns the factor for intracellular Na/K effect of piezo current
        """
        if isinstance(self.model_strategy, Piezo1DummyModel) or isinstance(self.model_strategy, SimplePiezo1Model) or isinstance(self.model_strategy, DoubleGateModel):
            return 0
        elif isinstance(self.model_strategy, MC4StateModel) and svr is not None and ratio_er is not None:
            factor = 0.10364
            return -factor * svr / (1 - ratio_er) 

        raise NotImplementedError("Na/K effect factor not implemented in Piezo1 channel!")
        


class Piezo1DummyModel(PiezoModelStrategy):
    
    def __init__(self) -> None:
        super().__init__("Piezo1DummyModel")

    def compute_currents(self, vm, t, mechanical_stimulus):
        return {
            'I_pz': 0
        }


class SimplePiezo1Model(PiezoModelStrategy):
    """
    Stretch-induced Mechanotransduction yielding a Ca flux.
    """

    def __init__(self, t_on_off: list[tuple[float, float]]) -> None:
        super().__init__("SimplePiezo1Model")

        self.G_pz = 45       # [nS]
        self.k = 1.2         # [nN/um]
        self.m = 5.7
        self.s = 0.68        # [nN^-1]
        self.k_b = 106
        self.tau_inact = 0.015  # [s]
        # self.E_rev = 0.006      # [V]
        self.E_rev = 6              # [mV]

        self.f_init = 1
        
        self.t_on_off = t_on_off

    def set_G_pz(self, G_pz):
        self.G_pz = G_pz
        
    def _d_pz(self, stretch):
        k_eff = self.k * stretch
        sigma_m = k_eff * stretch
        return 1 / (1 + self.k_b * math.exp(-self.s * (sigma_m / self.m)))
    
    def _f_pz(self, t):
        for t_on, t_off in self.t_on_off:
            if t_on <= t < t_off:
                return self.f_init * math.exp(-(t-t_on) / self.tau_inact)
            
        return 1

    def compute_currents(self, vm, t, mechanical_stimulus) -> dict[str, float]:
        # Model requires non-SI units -> convert:
        vm *= 1000
        mechanical_stimulus *= 10**6

        f_pz = self._f_pz(t)

        ## Original Model:
        # i_pz = self.G_pz * self._d_pz(mechanical_stimulus) * f_pz * (vm - self.E_rev)

        ## Refactored Model:
        # Calculate total stimulated open probability
        d_pz_total = self._d_pz(mechanical_stimulus)
        
        # Calculate the resting open probability (baseline leak when stretch = 0)
        d_pz_base = 1 / (1 + self.k_b) 
        
        # Isolate the mechanically active fraction
        d_pz_active = d_pz_total - d_pz_base
        
        # Apply inactivation only to the active fraction
        d_pz_effective = d_pz_base + (d_pz_active * f_pz)

        # Compute final current using the effective open probability
        i_pz = self.G_pz * d_pz_effective * (vm - self.E_rev)

        return {
            'I_pz': i_pz * 1e-12,   # convert back (kappa factor)
        }


class DoubleGateModel(PiezoModelStrategy):
    """
    Provides a tension-dependent activation gate and a voltage-dependent inactivation gate.
    """

    def __init__(self) -> None:
        super().__init__("DoubleGateModel")

        self.E_p = 0
        self.c = 0.116
        self.tau_a = 0.1
        self.g_p = 0.304

        self.xn = 0.5
        self.m_a = 0.5
      
        self.coeff_Q10_generico_tau = 1.

    def set_g_p(self, g_p):
        self.g_p = g_p

    def compute_currents(self, vm, t, mechanical_stimulus) -> dict[str, float]:
        i_pz = self.g_p * self.m_a * self.xn * (vm - self.E_p)

        return {
            'I_pz': i_pz,
        }
    
    def change_in_xn_ode(self, vm):
        xn_inf = 2.0 / (1.0 + math.exp(-vm*1000.0/16.0))
        alpha_xn = 1 / math.sqrt(1.0 + math.exp((-60 + vm*1000.0)/50.0))
        beta_xn = 270 / (1 + math.exp((171 - vm*1000.0)/72.0))
        tau_xn = alpha_xn * beta_xn
        # tau_xn      = self.coeff_Q10_generico_tau * 30.0*alpha_xn*beta_xn/100.0

        return (xn_inf - self.xn) / tau_xn
    
    def change_in_ma_ode(self, d):
        m_ainf = 1.0 / (1 + math.exp(-(d-2.5)/0.25))
        
        return (m_ainf - self.m_a) / self.tau_a
    

class MC4StateModel(PiezoModelStrategy):
    """
    Voltage-modulated 4 state Markov Chain Model.
    """

    # https://www.geeksforgeeks.org/dsa/finding-the-probability-of-a-state-at-a-given-time-in-a-markov-chain-set-2/
    # https://stackoverflow.com/questions/58048810/building-n-th-order-markovian-transition-matrix-from-a-given-sequence

    # https://en.wikipedia.org/wiki/Continuous-time_Markov_chain
    # https://vknight.org/blog/posts/continuous-time-markov-chains/

    def __init__(self, vm_0, p_0, pr_NaK) -> None:
        super().__init__("4-state MCM")

        # Init constants
        self.c1_e = -0.0064121
        self.c2_m = 0.124268
        self.c2_e = -0.0258118
        self.c3_e = 0.0142561
        self.c4_m = 0.0406628 
        self.c4_e = 0.0352718
        self.c5_e = 0.0182107
        self.c6_e = -0.0301922
        self.c7_e = 0.059666
        self.c8_e = -0.0082322
        self.r1 = 0.00504513
        self.r2 = 0.00043889
        self.r3 = 0.00160264
        self.r4 = 0.0182634
        self.r5 = 1.99342e-6
        self.r6 = 0.0132307 
        self.r7 = 9.2139e-5
        self.r8 = 0.0034847

        self.E_r = 0

        self.vm = vm_0
        self.p = p_0
        self.t_prev = 0.0

        self.thermal_voltage = constants.MOLAR_GAS_CONSTANT * simulation_config.TEMPERATURE / constants.FARADAY_CONSTANT  # RT / F
        self.pr_NaK = pr_NaK

        # Init Q and pi for Markov Chain
        self.states = ["O", "I_f", "I_s", "C"]

        self.Q = np.array(self._build_rate_matrix())
        self.pi = np.array([0, 0, 0, 1])

        # Init intra and extra
        self.intra: IntracellularSpace = None
        self.extra: ExtracellularSpace = None

        # # Init conductances
        # N_Pz1 = scaling * 52 * 10**6
        # g_bar_Ca = N_Pz1 * 12e-12
        # g_bar_K = N_Pz1 * 52e-12
        # g_bar_Na = N_Pz1 * 36e-12   # 10.1371/journal.pone.0125503
        # self.conductances_normalized = [g_bar_Na, g_bar_K, g_bar_Ca]

    def init_conductances(self, N_Pz1):
        g_bar_Ca = N_Pz1 * 12e-12
        g_bar_K = N_Pz1 * 52e-12
        g_bar_Na = N_Pz1 * 36e-12   # 10.1371/journal.pone.0125503
        self.conductances_normalized = [g_bar_Na, g_bar_K, g_bar_Ca]

    def _build_rate_matrix(self) -> list[list[float]]:
        """
        returns:
            Q -- rate matrix
        """
        delta_mu = self.vm - self.E_r

        transition_rates = [
            [0, self.r1 * np.exp(self.c1_e * delta_mu), self.r8 * np.exp(self.c8_e * delta_mu), self.r6 * np.exp(self.c6_e * delta_mu)],
            [self.r2 * np.exp(self.c2_m * self.p + self.c2_e * delta_mu), 0, 0, self.r3 * np.exp(self.c3_e * delta_mu)],                  
            [self.r7 * np.exp(self.c7_e * delta_mu * self.p), 0, 0, 0],                                                 
            [self.r5 * np.exp(self.c5_e * delta_mu), self.r4 * np.exp(self.c4_m * self.p + self.c4_e * delta_mu), 0, 0]
        ]

        for i in range(len(self.states)):
            transition_rates[i][i] = 0 - sum(transition_rates[i])
            # print(transition_rates[i])

        # N     T H E N
        # O
        # W
        # ################
        # 
        #      O     I_f     I_s     C
        # O
        # I_f
        # I_s
        # C

        return transition_rates
    
    def _update_state_probability(self, t):
        """
            dpi/dt = pi Q
            Analytical solution using matrix exponential
        """
        dt = (t - self.t_prev) * 1000
        if dt > 0:
            # only update when time is moving forward
            self.t_prev = t

            self.Q = np.array(self._build_rate_matrix())
            self.pi = self.pi @ sp.linalg.expm(self.Q * dt)
            
            # Prevent probabilities from drifting away from 1.0 over thousands of steps
            self.pi = self.pi / np.sum(self.pi)

    def compute_currents(self, vm, t, mechanical_stimulus) -> dict[str, float]:
        self.vm = vm
        self.p = mechanical_stimulus
        self._update_state_probability(t)

        p_O = self.pi[0]
        
        self._validate_intra_extra()

        E_Na = self.thermal_voltage * math.log(self.extra.sodium / self.intra.sodium)
        E_K = self.thermal_voltage * math.log((self.extra.potassium + self.pr_NaK * self.extra.sodium) / 
                                              (self.intra.potassium + self.pr_NaK * self.intra.sodium))
        if self.extra.calcium <= 0 or self.intra.calcium <= 0:
            print(f"{self.extra.calcium} / {self.intra.calcium}")
        E_Ca = self.thermal_voltage * 1/2 * math.log(self.extra.calcium / self.intra.calcium)
        
        nernst_potentials = [E_Na, E_K, E_Ca]

        currents = []
        for i, E in enumerate(nernst_potentials):
            g_bar = self.conductances_normalized[i]
            currents.append(p_O * g_bar * (vm - E))

        return {
            "I_Na": currents[0],
            "I_K": currents[1],
            "I_Ca": currents[2],
        }


    def _validate_intra_extra(self):
        if self.intra is None or self.extra is None:
            raise ValueError("Intra- and Extracellular Space must be configured in Piezo1 channel!")
