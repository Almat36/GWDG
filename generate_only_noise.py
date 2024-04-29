import matplotlib.pyplot as plt
import numpy as np
import pycbc.psd
import pycbc.noise

def generate_noise(sample_freq, duration, f_lower, plot=False, normalize=False):        
    # Calculate the Nyquist frequency
    nyquist_freq = sample_freq / 2

    # Calculate the frequency spacing
    delta_f_wv = nyquist_freq / sample_freq
    
    flen = int(sample_freq / delta_f_wv) + 1
    psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f_wv, f_lower)

    delta_f_psd = psd.delta_f
    delta_t = 1.0 / sample_freq
    tsamples = int(duration / delta_t)

    # Generate noise
    noise = pycbc.noise.gaussian.noise_from_psd(tsamples, delta_t, psd, seed=36)

    print('Length of noise: ', len(noise))

    if normalize == True:
        noise_min = min(noise)
        noise_max = max(noise)
        noise = -100 + (noise - noise_min) * (200 / (noise_max - noise_min))
    
    if plot == True:
        plt.figure(figsize=(10, 4))
        plt.plot(np.linspace(0, duration, tsamples), noise)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Strain')
        plt.title('Gaussian Noise')
        plt.show()

    return noise
