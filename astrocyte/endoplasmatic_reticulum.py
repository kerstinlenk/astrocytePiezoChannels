from utils.writable import Writable


class EndoplasmaticReticulum(Writable):
    
    _name = "EndoplasmaticReticulum"
    #NOTE: following constants not used because it is initialized in the config 
    # calcium = 20e-3 
    
    def __init__(self, w1):
        self._w1 = w1

    def change_in_calcium_ode(self, I_ip3, I_serca, I_caleak):
        return self._w1 * (-I_ip3 + I_serca - I_caleak)


    def get_name(self):
        return self._name
