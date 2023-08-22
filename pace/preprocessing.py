import wfdb
import numpy as np
from scipy import signal

"""
Author: Alice, Josue N Rivera
Date: 8/22/2023
"""

# Read record numbers
with open('./data/mitdb/RECORDS') as f:
    pat_ids = f.readlines()

for i in range(0, len(pat_ids)):
    pat_ids[i] = int(pat_ids[i])

# Categorize beat classes
classes_reduced = {'N':['N','L','R','e','j'],
                 'S':['S','A','a','J'],'V':['V','E'],'F':['F'],'Q':['/','Q','f']}
reduced = {}

for key, values in classes_reduced.items():
    for value in values:
        reduced[value] = key

segments = []
beat_types = []
valid_types = ['N','L','R','e','j','S','A','a','J','V','E','/','Q','f'] # Classified beats

for id in pat_ids:
    # Extract the ECG signal from the record
    record = wfdb.rdrecord(f'./data/mitdb/{id}')
    ecg_signal = record.p_signal[:,0]

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

    # Segment using relative minimum RR-interval
    annotation = wfdb.rdann(f'./data/mitdb/{id}', 'atr')
    loc = annotation.sample
    beat_type = annotation.symbol

    for i in range(2, len(loc)-1):
        if beat_type[i] not in valid_types:
            continue
        
        dist = round(min(loc[i]-loc[i-1], loc[i+1]-loc[i])*0.8)
        if dist * 2 < 100:
            continue

        segments.append(normalized_ecg_signal[loc[i]-dist:loc[i]+dist])
        beat_types.append(reduced[beat_type[i]])

# Padding
max_len = len(segments[0])
for segment in segments:
    max_len = max(len(segment), max_len)

for i in range(0, len(segments)):
    arr = np.zeros(max_len - len(segments[i]))
    segments[i] = np.concatenate((segments[i], arr))
