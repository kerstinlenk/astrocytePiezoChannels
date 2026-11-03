import numpy as np
from scipy import signal as sig

astrocyte_model = AstrocyteMorphologyModel()
astrocyte_1 = Astrocyte(geometry_path, spike_path, values_path, load_values,
                        simulate_morpology_model, svr, length, connection_matrix) # create the astrocyte without the connections

solution = astrocyte_1.solve() # computes the values over time

freq = []
for i,comp in enumerate(astrocyte_1._d):
    y_plot = astrocyte_1.y_plot
    peaks, _ = sig.find_peaks(solution.y[i*10+y_plot,:], threshold=1e-7)
    freq.append(len(peaks)/41.66)#41.66s is the length of the time window at the moment where we have the spike train
    
freq = np.round(np.array(freq)/0.24,2) #round the f so that they are in a range(0,1), important for the compartment color later
