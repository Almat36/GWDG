import h5py
import numpy as np

def from_hdf5(filename, load_params=False, load_waveform=False, load_noise=False, load_noisy_signals=False, load_entropy=False):
    data_dict = {}
    with h5py.File(filename, 'r') as f:
        if load_params:
            data_dict['params'] = {}
            for key in f.keys():
                if key.startswith('params_'):
                    subgroup = f[key]
                    data_dict['params'][key] = {}
                    for dataset_key in subgroup.keys():
                        data_dict['params'][key][dataset_key] = np.array(subgroup[dataset_key])

        if load_waveform:
            data_dict['waveforms'] = {}
            for key in f.keys():
                if key.startswith('waveforms_'):
                    subgroup = f[key]
                    data_dict['waveforms'][key] = {}
                    for dataset_key in subgroup.keys():
                        data_dict['waveforms'][key][dataset_key] = np.array(subgroup[dataset_key])

        if load_noise:
            data_dict['noises'] = {}
            for key in f.keys():
                if key.startswith('noises_'):
                    subgroup = f[key]
                    data_dict['noises'][key] = {}
                    for dataset_key in subgroup.keys():
                        data_dict['noises'][key][dataset_key] = np.array(subgroup[dataset_key])

        if load_noisy_signals:
            data_dict['signals'] = {}
            for key in f.keys():
                if key.startswith('signals_'):
                    subgroup = f[key]
                    data_dict['signals'][key] = {}
                    for dataset_key in subgroup.keys():
                        data_dict['signals'][key][dataset_key] = np.array(subgroup[dataset_key])

        if load_entropy:
            data_dict['entropies'] = {}
            for key in f.keys():
                if key.startswith('entropies_'):
                    subgroup = f[key]
                    data_dict['entropies'][key] = {}
                    for dataset_key in subgroup.keys():
                        data_dict['entropies'][key][dataset_key] = np.array(subgroup[dataset_key])

    return data_dict

data = from_hdf5('BBH_15_10.hdf5', load_noise=True, load_entropy=True)

#entropies = []
#for i in range(len(data['entropies'])):
