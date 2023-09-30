import numpy as np
from pace.db import get_patients_beats, cwt_parallel, pad_scalograms, get_label_distribution, split_train_test, hybrid_sample, under_sample
from pace import PATIENT_IDS

"""
Author: Alice Cheng, Josue N Rivera
Date: 8/22/2023
"""

def main():
    beats = []
    beat_IDs = []
    widths = np.arange(1, 31)
    data_path = 'data/mitdb/'

    for id in PATIENT_IDS: # take out patients for testing
        pat_beats, pat_beat_ID = get_patients_beats(ID=id, dt_path=data_path)

        beats.extend(pat_beats)
        beat_IDs.extend(pat_beat_ID)

    print("Completed data load")

    overall_dist = get_label_distribution(beat_IDs)
    train_dist, test_dist = split_train_test(overall_dist)
    beats_train, beat_IDs_train = hybrid_sample(beats, beat_IDs, train_dist)
    beats_test, beat_IDs_test = under_sample(beats, beat_IDs, test_dist)
    print("Completed data sampling")

    scalograms_train = cwt_parallel(beats=beats_train, widths=widths)
    scalograms_test = cwt_parallel(beats=beats_test, widths=widths)
    print("Completed cwt")

    scalograms_train = pad_scalograms(scalograms_train)
    scalograms_test = pad_scalograms(scalograms_test)
    print("Completed padding")

    scalograms_train = np.array(scalograms_train)
    scalograms_test = np.array(scalograms_test)
    np.savez_compressed("data/db_31_train.npz", scalograms=scalograms_train, labels=beat_IDs_train)
    np.savez_compressed("data/db_31_test.npz", scalograms=scalograms_test, labels=beat_IDs_test)
    print("Completed file save")

if __name__ == '__main__':
    main()
    