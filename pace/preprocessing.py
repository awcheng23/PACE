import wfdb
import numpy as np
import scipy as spy
from multiprocessing import Pool

"""
Author: Alice Cheng, Josue N Rivera
Date: 8/22/2023
"""

# Define a function to compute cwt for a single segment
widths = np.arange(1, 41)
def cwt_single_segment(segment):
    return spy.signal.cwt(segment, spy.signal.ricker, widths)

def cwt_parallel(segments):
        # Create a pool of worker processes
        pool = Pool()

        # Compute cwt for each segment in parallel
        cwt_data = pool.map(cwt_single_segment, segments)

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()

        # Return the result as a 3D array
        return np.array(cwt_data)

def main():
    # Read record numbers
    with open('./data/mitdb/RECORDS') as f:
        pat_ids = f.readlines()

    for i in range(0, len(pat_ids)):
        pat_ids[i] = int(pat_ids[i])

    # Categorize beat classes
    classes_reduced = {0:['N','L','R','e','j'], 1:['S','A','a','J'], 2:['V','E'], 3:['F'], 4:['/','Q','f']}
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
        b, a = spy.signal.butter(5, [low, high], btype='band')
        filtered_ecg_signal = spy.signal.filtfilt(b, a, ecg_signal)

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

    print("Completed data load")

    # Padding
    max_len = len(segments[0])
    for segment in segments:
        max_len = max(len(segment), max_len)

    for i in range(0, len(segments)):
        arr = np.zeros(max_len - len(segments[i]))
        segments[i] = np.concatenate((segments[i], arr))

    print("Completed padding")

    # Produce scalogram in parallel
    segments = cwt_parallel(segments)

    print("Completed cwt")

    segments = np.array(segments)
    np.savez("./data/db.npz", segments=segments, labels=beat_types)

    print("Completed file save")

if __name__ == '__main__':
    main()