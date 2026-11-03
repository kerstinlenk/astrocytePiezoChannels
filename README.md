# Piezo1 channel integration in computational astrocyte model: simulation code
Single-compartment astrocyte model in its updated version extended by different Piezo1 channel implementations.

## Piezo1 architectures
- **P1.dual:** Simple dual-gate Piezo1 model (Model 1).
- **P1.dual-vm:** Dual-gate Piezo1 model with voltage-dependent inactivation (Model 2).
- **P1.mc-vm:** Voltage-modulated Piezo1 Markov chain model (Model 3). 
- **Reference:** No mechanotransduction; no additional ion flux, i.e. $I_{pz} = 0$ A. For reference purposes (Dummy model).

## Run the code
To run the simulation/analyses execute the `main.py` script.

Using the option `rerun=False`, you will recreate all plots and analyses without rerunning the simluation. The csv-datafiles from the current release must be provided in the correct structure anchored in the project folder: 

```text
output/
├── model_1/
│   ├── all_currents_vm.csv
│   ├── result.csv
│   └── spikes/
├── model_2/
│   ├── all_currents_vm.csv
│   ├── result.csv
│   └── spikes/
├── model_3/
│   ├── all_currents_vm.csv
│   ├── result.csv
│   └── spikes/
└── model_dummy/
    ├── all_currents_vm.csv
    └── result.csv
```

With `rerun=True` you can additionally rerun the simluation and subsequently create all plots and analyses.
