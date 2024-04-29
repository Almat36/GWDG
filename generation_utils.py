from pycbc.waveform import get_td_waveform
from pycbc.filter import highpass_fir
from pycbc.detector import Detector
from pycbc.types import TimeSeries
import matplotlib.pyplot as plt
from pycbc.filter import sigma
from lal import LIGOTimeGPS
import antropy as ant
import random as rand
import pycbc.noise
import numpy as np
import pycbc.psd
import math
import h5py
import json

def get_param_set(sim_params):
    """
    Generate a set of parameters for a simulated gravitational wave signal based on input specifications.

    Parameters:
        sim_params (dict): Dictionary containing simulation parameters such as number of signals, mass range, 
                           spin range, and others specific to different classes of compact binary coalescences.

    Returns:
        dict: A dictionary with generated parameters for a single gravitational wave signal.
    """
    param_set = {}
    for i in range(0, sim_params['num_signals']):
        param_set['CBC_class'] = sim_params['CBC_class']
        param_set['m1'] = rand.uniform(sim_params['mass_range'][0], sim_params['mass_range'][1])
        if sim_params['CBC_class'] == 'NSBH':
            param_set['m2'] = rand.uniform(1, 2)
        else:      
            param_set['m2'] = rand.uniform(sim_params['mass_range'][0], sim_params['mass_range'][1])

	#Stick to convention, make the larger body mass1
        if param_set['m2']>param_set['m1']:
            m_lesser = param_set['m1']
            param_set['m1']=param_set['m2']
            param_set['m2']=m_lesser
			
        param_set['apx'] = sim_params['approximant']
        param_set['x1'] = rand.uniform(sim_params['spin_range'][0], sim_params['spin_range'][1])
        param_set['x2'] = rand.uniform(sim_params['spin_range'][0], sim_params['spin_range'][1])
        param_set['inc'] = rand.uniform(sim_params['inclination_range'][0], sim_params['inclination_range'][1])
        param_set['coa'] = rand.uniform(sim_params['coa_phase_range'][0], sim_params['coa_phase_range'][1])
        param_set['ra'] = rand.uniform(sim_params['right_asc_range'][0], sim_params['right_asc_range'][1])
        param_set['dec'] = math.asin(1-(2*rand.uniform(sim_params['declination_range'][0], sim_params['declination_range'][1])))
        param_set['pol'] = rand.uniform(sim_params['polarisation_range'][0], sim_params['polarisation_range'][1])
        param_set['dist'] = rand.randint(sim_params['distance_range'][0], sim_params['distance_range'][1])
        param_set['snr'] = 10**(sim_params['snr_db']/10)
        param_set['sf'] = sim_params['sample_frequency']
		
    yield param_set

def normalize_signal(signal, minimum, maximum):
    """
    Normalize the input signal to a specified range.

    Parameters:
        signal (array-like): The input signal to be normalized.
        minimum (float): The minimum value of the normalized range.
        maximum (float): The maximum value of the normalized range.

    Returns:
        array: The normalized signal.
    """
    signal_min = min(signal)
    signal_max = max(signal)
    signal_norm = minimum + (signal - signal_min) * (2 * maximum / (signal_max - signal_min))

    return signal_norm

def resize_signal(hp,seconds_before_event, seconds_after_event, sample_freq):
    """
    Resize a signal around its peak to specified seconds before and after the peak event.

    Parameters:
        hp (array-like): The input signal (time series).
        seconds_before_event (int): Seconds to keep before the peak event.
        seconds_after_event (int): Seconds to keep after the peak event.
        sample_freq (float): Sampling frequency of the signal.

    Returns:
        array: The resized signal.
    """
    signal_len_t = seconds_before_event + seconds_after_event         
    signal_len = int(signal_len_t * sample_freq)   

    peak_index = np.argmax(hp)
    cut_start = peak_index - int(seconds_before_event * sample_freq)
    
    if len(hp) >= signal_len:
        hp_cut = hp[cut_start : cut_start + signal_len]
    else:
        pad_length = signal_len - len(hp)
        left_pad = abs(cut_start)
        right_pad = signal_len - left_pad - len(hp)
        hp_padded = np.pad(hp, (left_pad, abs(right_pad)), 'constant')
        hp_cut = hp_padded[:signal_len]  

    # If the cut signal is shorter than needs to be, fill the remaining with zeros
    if len(hp_cut) < signal_len:
        hp_cut = np.pad(hp_cut, (0, signal_len - len(hp_cut)), 'constant')
        
    return hp_cut
    
def generate_waveform(param_list, plot = False):
    """
    Generate gravitational waveforms for detectors H1 and L1 based on simulation parameters,
    and optionally plot these waveforms.

    Parameters:
        param_list (dict): Dictionary containing the parameters necessary to generate waveforms,
                           including masses, spins, distance, etc.
        plot (bool, optional): If True, plot the generated waveforms.

    Returns:
        dict: Dictionary containing the generated waveforms for detectors 'H1' and 'L1'.
    """
    seconds_before_event = 4
    seconds_after_event = 2     
    
    hp, hc = get_td_waveform(approximant = param_list['apx'],
                                    mass1 = param_list['m1'],
				    mass2 = param_list['m2'],
				    spin1z = param_list['x1'],
				    spin2z = param_list['x2'],
				    inclination_range = param_list['inc'],
				    coa_phase = param_list['coa'],
				    distance = param_list['dist'],
				    delta_t = 1.0 / param_list['sf'],
				    f_lower = 20)

    det_h1 = Detector('H1')
    det_l1 = Detector('L1')

    waveform_h1 = det_h1.project_wave(hp, hc, param_list['ra'], param_list['dec'], param_list['pol'])
    waveform_l1 = det_l1.project_wave(hp, hc, param_list['ra'], param_list['dec'], param_list['pol'])
	
    hp_h1 = resize_signal(waveform_h1, seconds_before_event, seconds_after_event, param_list['sf'])
    hp_l1 = resize_signal(waveform_l1, seconds_before_event, seconds_after_event, param_list['sf'])

    hp_h1 = normalize_signal(hp_h1, -100, 100)
    hp_l1 = normalize_signal(hp_l1, -100, 100)

    sig_h1 = TimeSeries(hp_h1, delta_t=1.0/param_list['sf'])
    sig_l1 = TimeSeries(hp_l1, delta_t=1.0/param_list['sf'])
                          
    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)    

        axs[0].plot(sig_h1.sample_times, sig_h1, color='royalblue', label='Hanford')
        axs[0].set_ylabel('Strain')
        axs[0].set_xlim(2, 6)
        #axs[0].set_xticks(np.arange(0, 4.1, 0.5))
        axs[0].legend()

        axs[1].plot(sig_l1.sample_times, sig_l1, color='firebrick', label='Livingston')
        axs[1].set_ylabel('Strain')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_xlim(2, 6)
        #axs[1].set_xticks(np.arange(0, 4.1, 0.5))
        axs[1].legend()

        #plt.show()

    return {'H1':sig_h1, 'L1':sig_l1}

def inject_signal(waveform_dict, inj_snr, param_list, plot=False):
    """
    Inject generated waveforms into noise to simulate real detector data, calculate signal-to-noise
    ratio (SNR), and optionally plot these signals.

    Parameters:
        waveform_dict (dict): Dictionary containing the waveforms for 'H1' and 'L1'.
        inj_snr (float): Target network SNR for the injection.
        param_list (dict): Dictionary containing the simulation parameters.
        plot (bool, optional): If True, plot the noisy and original waveforms.

    Returns:
        tuple: Tuple containing dictionaries of noisy signals and noise for each detector.
    """
    noise = dict()
    for i,det in enumerate(('H1', 'L1')):
        f_low = 20
        delta_f = waveform_dict[det].delta_f
        flen = int(param_list['sf'] / delta_f) + 1
        psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, f_low)
        noise[det] = pycbc.noise.gaussian.noise_from_psd(length=param_list['sf']*8,
                                                        delta_t=1.0/param_list['sf'],
                                                        psd=psd)
        
        noise[det]._epoch = LIGOTimeGPS(waveform_dict[det].start_time)
        noise[det] = normalize_signal(noise[det], -100, 100)
        
    psds = dict()
    dummy_strain = dict()
    snrs = dict()

    #using dummy strain and psds from the noise, calculate the snr of each signal+noise injection to find the 
    #network optimal SNR, used for injecting the real signal
    for det in ('H1', 'L1'):
        delta_f = waveform_dict[det].delta_f
        dummy_strain[det] = noise[det].add_into(waveform_dict[det])
        psds[det] = dummy_strain[det].psd(8)
        psds[det] = pycbc.psd.interpolate(psds[det], delta_f=delta_f)
        snrs[det] = sigma(htilde=waveform_dict[det],
                                        psd=psds[det],
                                        low_frequency_cutoff=f_low)
        
    nomf_snr = np.sqrt(snrs['H1']**2 + snrs['L1']**2)
    scale_factor = inj_snr / nomf_snr
    noisy_signals = dict()
    
    for det in ('H1', 'L1'):
        noisy_signals[det] = noise[det].add_into(waveform_dict[det]*scale_factor)

	#Whiten and bandpass signal
        noisy_signals[det] = noisy_signals[det].whiten(segment_duration = 0.001,
                                                        max_filter_duration = 8, 
                                                        remove_corrupted = False,
                                                        low_frequency_cutoff = f_low)
      
        noisy_signals[det] = noisy_signals[det].highpass_fir(frequency = f_low ,remove_corrupted=False, order=8)
        noisy_signals[det] = normalize_signal(noisy_signals[det], -100, 100)
        
        waveform_dict[det] = waveform_dict[det]*scale_factor
        waveform_dict[det] = normalize_signal(waveform_dict[det], -100, 100)

	#Cut down to desired length and cut off corrupted tails of signal
        if param_list['CBC_class'] == 'BNS':
            noisy_signals[det] = noisy_signals[det].time_slice(-3, 1)
            waveform_dict[det] = waveform_dict[det].time_slice(-3, 1)
        else:    
            noisy_signals[det] = noisy_signals[det].time_slice(1, 5)
            waveform_dict[det] = waveform_dict[det].time_slice(1, 5)

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex=True)   

        axs[0].plot(noisy_signals['H1'].sample_times, noisy_signals['H1'], color='royalblue', label='Noisy signal')
        axs[0].plot(waveform_dict['H1'].sample_times, waveform_dict['H1'], color='firebrick', label='Waveform')
        axs[0].set_ylabel('Strain')

        axs[1].plot(noisy_signals['L1'].sample_times, noisy_signals['L1'], color='royalblue')
        axs[1].plot(waveform_dict['L1'].sample_times, waveform_dict['L1'], color='firebrick')
        axs[1].set_ylabel('Strain')
        axs[1].set_xlabel('Time (s)')

        fig.legend(loc='upper right')

        fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex=True)   

        axs[0].plot(noisy_signals['H1'].sample_times, noisy_signals['H1'], color='royalblue', label='Noisy signal')
        axs[0].plot(waveform_dict['H1'].sample_times, waveform_dict['H1'], color='firebrick', label='Waveform')
        axs[0].set_ylabel('Strain')
        axs[0].set_xlim(3.95, 4.02)

        axs[1].plot(noisy_signals['L1'].sample_times, noisy_signals['L1'], color='royalblue')
        axs[1].plot(waveform_dict['L1'].sample_times, waveform_dict['L1'], color='firebrick')
        axs[1].set_ylabel('Strain')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_xlim(3.95, 4.02)

        fig.legend(loc='upper right')
        #plt.show()
    
    return noisy_signals, noise

def entropy(data, window_size, step, sf, duration, plot=True):
    """
    Calculate the spectral entropy of data for given parameters and optionally plot the entropy.

    Parameters:
        data (dict): Dictionary containing data for 'H1' and 'L1'.
        window_size (int): Size of the window for calculating entropy.
        step (int): Step size between windows.
        sf (int): Sampling frequency.
        duration (int): Total duration of data to be analyzed.
        plot (bool, optional): If True, plot the entropy values.

    Returns:
        dict: Dictionary containing the entropy values for each detector.
    """
    entropy_values = {det: [] for det in ('H1', 'L1')} 
    
    for det in ('H1', 'L1'):
        j = 0
        for k in range(window_size, sf * duration, step):
            entropy_values[det].append(ant.spectral_entropy(data[det][j:k], sf=sf, method='fft', normalize=True))
            j += 1

    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(np.linspace(0, duration, len(entropy_values[det])), entropy_values['H1'], 'k')
        plt.show()
        
    return entropy_values

def convert_ts_to_np(timeseries_list):
    """
    Convert a list of dictionaries containing TimeSeries objects into numpy arrays.

    Parameters:
        timeseries_list (list): List of dictionaries where each dictionary contains TimeSeries objects.

    Returns:
        list: List of dictionaries where TimeSeries objects are replaced with numpy arrays.
    """
    data_arrays = []
    for item in timeseries_list:
        data_arrays.append({
            key: np.array(value.data) if isinstance(value, pycbc.types.timeseries.TimeSeries) else value
            for key, value in item.items()
        })
    return data_arrays

def to_hdf5(filename, param_list, waveform_list=None, noise_list=None, noisy_signal_list=None, 
                 entropy_list=None, save_params=False, save_waveform=False, save_noise=False, 
                 save_noisy_signals=False, save_entropy=False):
    """
    Save various types of data to an HDF5 file, with options to save parameters, waveforms, noise profiles, 
    noisy signals, and entropy calculations.

    Parameters:
        filename (str): The name of the HDF5 file to write the data to.
        param_list (list): List of dictionaries containing parameters for each simulation.
        waveform_list (list, optional): List of dictionaries containing waveform data.
        noise_list (list, optional): List of dictionaries containing noise data.
        noisy_signal_list (list, optional): List of dictionaries containing noisy signal data.
        entropy_list (list, optional): List of dictionaries containing entropy data.
        save_params (bool): If True, save parameters to the HDF5 file.
        save_waveform (bool): If True, save waveforms to the HDF5 file.
        save_noise (bool): If True, save noise data to the HDF5 file.
        save_noisy_signals (bool): If True, save noisy signals to the HDF5 file.
        save_entropy (bool): If True, save entropy data to the HDF5 file.

    """
    with h5py.File(filename, 'w') as f:
        if save_params:
            params = convert_ts_to_np(param_list)
            for i, data_dict in enumerate(param_list):
                group = f.create_group(f'params_{i}')
                for key, value in data_dict.items():
                    group.create_dataset(key, data=value)

        if save_waveform:
            waveforms = convert_ts_to_np(waveform_list)
            for i, data_dict in enumerate(waveform_list):
                group = f.create_group(f'waveforms_{i}')
                for key, value in data_dict.items():
                    group.create_dataset(key, data=value)

        if save_noise:
            noises = convert_ts_to_np(noise_list)
            for i, data_dict in enumerate(noise_list):
                group = f.create_group(f'noises_{i}')
                for key, value in data_dict.items():
                    group.create_dataset(key, data=value)

        if save_noisy_signals:
            noisy_signals = convert_ts_to_np(noisy_signal_list)
            for i, data_dict in enumerate(noisy_signal_list):
                group = f.create_group(f'signals_{i}')
                for key, value in data_dict.items():
                    group.create_dataset(key, data=value)

        if save_entropy:
            entropies = convert_ts_to_np(entropy_list)
            for i, data_dict in enumerate(entropy_list):
                group = f.create_group(f'entropies_{i}')
                for key, value in data_dict.items():
                    group.create_dataset(key, data=value)

  




                       







   
