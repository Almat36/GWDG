# GWDG
This repository contains Python scripts for generating synthetic gravitational wave signals and adding noise to simulate realistic observations. The generated signals are saved in HDF5 format for further analysis.

# Contents:

generate_signals.py:
The main script for generating gravitational wave signals.

generation_utils.py:
Contains utility functions for signal generation and manipulation.
Includes functions for injecting signals, generating waveforms, calculating entropy, and normalizing signals.

Generated data saved in HDF5 format. HDF5 file contains the generated gravitational wave signals and associated metadata.
Signals are stored along with corresponding parameters, waveforms, noise, and entropy calculations.

#Quickstart:
As of now, this project is not a "proper" Python package, but only a collection of scripts, which don't need to be installed. Therefore, simply clone the repository:
```
git clone https://github.com/Almat36/GWDG.git
```
Ensure your Python environment fulfills all the requirements specified in requirements.txt. Please note that due to the dependence on PyCBC, this code currently only works in Linux/MacOS. Now, you should be able to generate your first data sample by simply running:
```
python3 generate_dataset.py
```
The script supports different classes of compact binary coalescence (CBC) events, such as BNS (binary neutron star), NSBH (neutron star black hole), and BBH (binary black hole).
Signals are generated with specified signal-to-noise ratio (snr) and other parameters such as mass range, spin range, and inclination range.
Progress is displayed using a progress bar during signal generation.

# Authors & License
The code in this repository is developed by Almat Akhmetali. It is distributed under the GPL-3.0 license, which means in particular that it is provided as is, without any warranty, liability, or guarantees regarding its correctness.
