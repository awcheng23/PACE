import wfdb
import numpy as np
from scipy import signal

## To get started

# Read a record from the database
record = wfdb.rdrecord('./data/mitdb/100')

# Extract the ECG signal from the record
ecg_signal = record.p_signal[:,1]

# Apply a bandpass filter to remove noise
lowcut = 0.5
highcut = 45.0
fs = record.fs
nyquist = 0.5 * fs
low = lowcut / nyquist
high = highcut / nyquist
b, a = signal.butter(5, [low, high], btype='band')
filtered_ecg_signal = signal.filtfilt(b, a, ecg_signal)

# Normalize the signal
min_signal, max_signal = np.min(filtered_ecg_signal), np.max(filtered_ecg_signal) 
normalized_ecg_signal = (filtered_ecg_signal - min_signal) / (max_signal - min_signal)

print(normalized_ecg_signal)
print(len(normalized_ecg_signal))