import numpy as np
from pace.db import get_patients_beats, cwt_parallel, padd_scalograms
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

    for id in PATIENT_IDS:
        pat_beats, pat_beat_ID = get_patients_beats(ID=id, dt_path=data_path)

        beats.extend(pat_beats)
        beat_IDs.extend(pat_beat_ID)

    print("Completed data load")

    scalograms = cwt_parallel(beats=beats, widths=widths)
    print("Completed cwt")

    scalograms = padd_scalograms(scalograms)
    print("Completed padding")

    scalograms = np.array(scalograms)
    np.savez_compressed("data/db_compressed.npz", scalograms=scalograms, labels=beat_IDs)
    print("Completed file save")

if __name__ == '__main__':
    main()