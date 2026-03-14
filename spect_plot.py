import numpy as np
import matplotlib.pyplot as plt
# Generate some sample speech signals
from scipy.io import wavfile
from scipy.signal import decimate
sample_rate1,speech1=wavfile.read('E:\phd_proj_with_preprocc\cat_6.wav')
speech1= np.array(speech1)
min_value1 = np.min(speech1)
max_value1 = np.max(speech1)
nspeech1 = (speech1 - min_value1) / (max_value1 - min_value1)
sample_rate2,speech2=wavfile.read('E:\phd_proj_with_preprocc\dog_13.wav')
speech2= np.array(speech2)
min_value2 = np.min(speech2)
max_value2 = np.max(speech2)
nspeech2 = (speech2 - min_value2) / (max_value2 - min_value2)
sample_rate3,speech3=wavfile.read('E:\phd_proj_with_preprocc\doggrowl2.wav')
sample_rate3=sample_rate3/3
speech3=decimate(speech3,3)
##
speech3= np.array(speech3)
min_value3 = np.min(speech3)
max_value3 = np.max(speech3)
nspeech3 = (speech3 - min_value3) / (max_value3 - min_value3)
##
sample_rate4,speech4=wavfile.read('E:\phd_proj_with_preprocc\Dog2grunt.wav')
sample_rate4=sample_rate4/3
speech4=decimate(speech4,3)
##
speech4= np.array(speech4)
min_value4 = np.min(speech4)
max_value4 = np.max(speech4)
nspeech4 = (speech4 - min_value4) / (max_value4 - min_value4)
##
# Create a figure and axis
fig, ax = plt.subplots()
# Create a box plot for each signal
ax.boxplot([nspeech1, nspeech2, nspeech3, nspeech4], labels=['Other Animal', 'Dog bark', 'Dog growl', 'Dog grunt'])
# Set the title and labels
ax.set_title('Box Plot of various Dog barks')
ax.set_xlabel('Signal')
ax.set_ylabel('Amplitude')

# Show the plot
plt.show()
