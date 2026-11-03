from utils.writable import Writable


class IntracellularSpace(Writable):
    _name = "IntracellularSpace"
    #NOTE: following constants not used because it is initialized in the config
    # calcium = 7.3e-5
    # sodium = 15
    # potassium = 100
    # ip3 = 0.16e-3
    
    def __init__(self, w1, w2, w34, w5):
        self._w1 = w1
        self._w2 = w2
        self._w34 = w34
        self._w5 = w5

    def get_name(self):
        return self._name

    def change_in_calcium_ode(self, I_ncx, I_ip3, I_serca, I_caleak, I_pz_Ca):
        # return self._w1 * 0.5 * I_ncx + self._w2 * (I_ip3 - I_serca + I_caleak) + self._w34 * I_Ca 
        return self._w1 * 0.5 * I_ncx + self._w2 * (I_ip3 - I_serca + I_caleak) + self._w34 * I_pz_Ca

    def change_in_potassium_ode(self, I_gluT, I_nka, I_leak, I_pz_K=0):
        # return self._w1 * (-I_gluT + 2 * I_nka - I_leak) - self._w34 * I_K
        return self._w1 * (-I_gluT + 2 * I_nka - I_leak) + self._w5 * I_pz_K
        # return self._w1 * (-I_gluT + 2 * I_nka - I_leak) - self._w1 * abs(I_K)

    def change_in_sodium_ode(self, I_gluT, I_nka, I_ncx, I_leak, I_pz_Na=0):
        # return self._w1 * (3 * I_gluT - 3 * I_nka - 3 * I_ncx - I_leak) + self._w34 * I_Na
        return self._w1 * (3 * I_gluT - 3 * I_nka - 3 * I_ncx - I_leak) + self._w5 * I_pz_Na

    @staticmethod #NOTE: why is this static?
    def change_in_ip3_ode(plc_beta, plc_delta, ip33k, ip5p):
        return plc_beta + plc_delta - ip33k - ip5p
