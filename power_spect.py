import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import decimate
sample_rate,speech=wavfile.read('E:\phd_proj_with_preprocc\dog_grunt_train\Dog3.wav')
sample_rate=sample_rate/3
speech=decimate(speech,3) #8 order chebyshev type-1 filter
fft_result = np.fft.fft(speech)

# Calculate the power spectral density
power_spectral_density = np.abs(fft_result)**2

# Plot the power spectral density
freq = np.linspace(0, sample_rate/2, len(power_spectral_density)//2)
plt.plot(freq, power_spectral_density[0:len(freq)])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density')
plt.title('Dog grunt')
plt.show()