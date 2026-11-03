from utils.writable import Writable


class ExtracellularSpace(Writable):

    _name = "ExtracellularSpace"

    #NOTE: following constants not used because it is initialized in the config
    # calcium = 1.800657  # Ca^2+
    # sodium = 150  # Na
    # potassium = 3
    # glu = 0.002  # NOTE - This is basically the external stimulus.


    def get_name(self):
        return self._name
