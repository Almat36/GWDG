# GWDG
This repository contains Python scripts for generating synthetic gravitational wave signals and adding noise to simulate realistic observations. The generated signals are saved in HDF5 format for further analysis.

# Contents:

Main Script (generate_signals.py):
The main script for generating gravitational wave signals.

Utility Module (generation_utils.py):
Contains utility functions for signal generation and manipulation.
Includes functions for injecting signals, generating waveforms, calculating entropy, and normalizing signals.

Generated Data File ({CBC_class}_{snr}_{num_signals}.hdf5):
HDF5 file containing the generated gravitational wave signals and associated metadata.
Signals are stored along with corresponding parameters, waveforms, noise, and entropy calculations.

#Usage:

Setup:
Ensure Python and required dependencies (matplotlib, numpy, progressbar) are installed.
Import generation_utils.py for utility functions.
Execution:
Modify parameters such as CBC_class, snr, and num_signals as needed.
Run the main script (generate_signals.py) to generate synthetic signals.
Signals are saved in HDF5 format for further analysis.
Output:
HDF5 file ({CBC_class}_{snr}_{num_signals}.hdf5) containing generated signals and metadata.
Notes:

The script supports different classes of compact binary coalescence (CBC) events, such as BNS (binary neutron star), NSBH (neutron star black hole), and BBH (binary black hole).
Signals are generated with specified signal-to-noise ratio (snr) and other parameters such as mass range, spin range, and inclination range.
Progress is displayed using a progress bar during signal generation.

# Authors & License
The code in this repository is developed by Almat Akhmetali. It is distributed under the GPL-3.0 license, which means in particular that it is provided as is, without any warranty, liability, or guarantees regarding its correctness.
