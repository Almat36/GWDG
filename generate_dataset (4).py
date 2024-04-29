'''
	Almat Akhmetali 2024
	start: 24.04.24
'''

import matplotlib.pyplot as plt
import numpy as np
from generation_utils import inject_signal, generate_waveform, entropy, normalize_signal
from generation_utils import get_param_set, convert_ts_to_np, to_hdf5 
import time
from progressbar import ProgressBar

CBC_class = 'BBH'
snr = 15
num_signals = 50
filename = f"{CBC_class}_{snr}_{num_signals}.hdf5"

if CBC_class == 'BNS':
    mass_range = (1,2)
    appx = 'TaylorF2'
elif CBC_class == 'NSBH':
    mass_range = (2,35)
    appx = 'IMRPhenomNSBH'
else:
    mass_range = (10,80)
    appx = 'SEOBNRv4'
    
sim_params = {
        'CBC_class': CBC_class,
	'mass_range':mass_range,
        'approximant': appx,
        'snr_db':snr,
        'num_signals':num_signals,
	'spin_range':(-1,1),
	'inclination_range':(0, np.pi),
	'coa_phase_range':(0, 2*np.pi),
	'right_asc_range':(0, 2*np.pi),
	'declination_range':(0, 1),
	'polarisation_range':(0, 2*np.pi),
	'distance_range':(40, 3000),
	'sample_frequency':4096,
}

param_list = []
waveform_list = []
noise_list = []
noisy_signal_list = []
entropy_list = []

progress_bar = ProgressBar(num_signals)

for i in range(num_signals):
    start = time.time()
    if i == 0:
        print(f"Starting generation of {sim_params['num_signals']} signals ...")

    param_list.append(next(get_param_set(sim_params)))
    waveform_list.append(generate_waveform(param_list[-1], plot=False))
    noisy_signal, noise = inject_signal(waveform_list[-1], param_list[-1]['snr'], param_list[-1], plot=False)
    noisy_signal_list.append(noisy_signal)
    noise_list.append(noise)
    entropy_list.append(entropy(noisy_signal_list[-1], 1024, 1, param_list[-1]['sf'], 4, plot=False))

    progress_bar.update(i + 1)
    
    #Every x loops, save the samples generated, stops memory errors when generating large datasets
    x=1000
    if i%x==0 and i!=0:
        print('Finished {0}...'.format(i))
        to_hdf5(filename, param_list, waveform_list, noise_list, noisy_signal_list, entropy_list,
                   save_noise=True, save_entropy=True)
        param_list = []
        waveform_list = []
        noise_list = []
        noisy_signal_list = []
        entropy_list = []
    
progress_bar.finish()

to_hdf5(filename, param_list, waveform_list, noise_list, noisy_signal_list, entropy_list,
        save_noise=True, save_entropy=True)

end = time.time()
print('\nFinished! Took {0} seconds to generate and save {1} samples.\n'.format(float(end-start), sim_params['num_signals']))
