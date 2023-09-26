import numpy as np
from pace.db import get_patients_beats, cwt_parallel, pad_scalograms
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

    # label distribution dist
    dist={}

    # split training and test per lab (e.g., 80% 20%); dict for each dataset

    training_dist = {}
    test_dist = {}
    for key in dist:
        split = int(len(dist['key'])*0.8)
        training_dist[key] = dist['key'][:split]
        test_dist[key] = dist['key'][split:]

    # uniform fold training
    training_dist

    # cap each label
    test_dist

    scalograms = cwt_parallel(beats=beats, widths=widths)
    print("Completed cwt")

    scalograms = pad_scalograms(scalograms, 966)
    print("Completed padding")



    scalograms = np.array(scalograms)
    np.savez_compressed("data/db.npz", scalograms=scalograms, labels=beat_IDs)
    print("Completed file save")

if __name__ == '__main__':
    main()
    