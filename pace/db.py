from typing import Optional, Tuple, List, Union
import numpy as np
import torch as th
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import wfdb
import numpy as np
import scipy as spy
from multiprocessing import Pool
from os.path import join as join_path
from pace import BEAT_TO_ID

"""
Author: Alice Cheng, Josue N Rivera
Date: 8/22/2023
"""

valid_types = ['N','L','R','e','j','S','A','a','J','V','E','F','/','Q','f'] # Classified beats

def get_record(ID:int, dt_path: str = 'data/mitdb/'):

    """Obtain a patient record"""

    path = join_path(dt_path, f'{ID}')
    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, 'atr')

    return record, annotation

def get_bandpass_filter_signal(record:wfdb.Record, 
        lowcut:float = 0.5,
        highcut:float = 45.0) -> np.ndarray:
    
    """Apply a bandpass filter to remove noise and get signal"""

    ecg_signal = record.p_signal[:,0]
    fs = record.fs
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = spy.signal.butter(5, [low, high], btype='band')

    return spy.signal.filtfilt(b, a, ecg_signal)

def normalize_signal(signal:np.ndarray,
                  min_signal:Optional[float] = None,
                  max_signal:Optional[float] = None) -> np.ndarray:
    
    """Normalize a signal to the range [0, 1]"""

    min_signal = np.min(signal) if type(min_signal) == type(None) else min_signal
    max_signal = np.max(signal) if type(max_signal) == type(None) else max_signal

    return (signal - min_signal) / (max_signal - min_signal)

def segment_signal_relative(signal:np.ndarray,
                            annotation: wfdb.Annotation,
                            relative_ratio:float = 0.8,
                            low_threshold:int = 100,
                            high_threshold:int = 1000) -> Tuple[List[np.ndarray], List[int]]:
    
    """Segment signal using relative minimum RR-interval"""

    loc = annotation.sample
    beat_type = annotation.symbol

    beats = []
    beat_IDs = []

    for i in range(2, len(loc)-1):
        if beat_type[i] not in valid_types:
            continue
        
        dist = round(min(loc[i]-loc[i-1], loc[i+1]-loc[i])*relative_ratio)
        if dist * 2 < low_threshold or dist * 2 > high_threshold:
            continue

        beats.append(signal[loc[i]-dist:loc[i]+dist])
        beat_IDs.append(BEAT_TO_ID[beat_type[i]])

    return beats, beat_IDs

def pad_scalograms(scalograms:Union[list, np.ndarray], max_length: Optional[int] = None):

    """Pad scalograms to the same size"""

    scalograms = [scalograms] if type(scalograms) == np.ndarray else scalograms

    max_length = max([scalogram.shape[1] for scalogram in scalograms]) \
                    if type(max_length) == type(None) else max_length
   
    for i in range(0, len(scalograms)):
        padding = ((0,0), (0,max_length-scalograms[i].shape[1]))
        scalograms[i] = np.pad(scalograms[i], pad_width=padding, mode='constant', constant_values=0)

    return scalograms
    
def cwt_single_beat(beat: np.ndarray, widths: np.ndarray) -> np.ndarray:

    """Create a single scalogram"""

    return spy.signal.cwt(beat, spy.signal.ricker, widths)
    
def _cwt_single_beat(args):

    """Create a single scalogram"""

    beat, widths = args
    return spy.signal.cwt(beat, spy.signal.ricker, widths)

def cwt_parallel(beats: List[np.ndarray], widths: np.ndarray, processes:int = 2) -> List[np.ndarray]:

    """Create scalograms in parallel"""

    # Create a pool of worker processes
    pool = Pool(processes)

    # Compute cwt for each segment in parallel
    cwt_data = pool.map(_cwt_single_beat, [(beat, widths) for beat in beats])

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    # Return the result as a 3D array
    return cwt_data

def get_patients_beats(ID:int, dt_path: str = 'data/mitdb/') -> Tuple[List[np.ndarray], List[int]]:

    """Get all beats for a patient"""

    record, annotation = get_record(ID=ID, dt_path=dt_path)
    signal = get_bandpass_filter_signal(record=record)
    signal = normalize_signal(signal=signal)
    beats, beat_IDs = segment_signal_relative(signal=signal, annotation=annotation)

    return beats, beat_IDs

def uniform_sampling(beats: List[np.ndarray],
                     beat_IDs: List[int], 
                     num_samples:int = 2763) -> Tuple[List[np.ndarray], List[int]]:
    
    """Sample uniformly from each beat class"""

    # Form distribution list for each beat class
    id_split = {}
    beat_IDs_unique = list(set(beat_IDs))
    for id in beat_IDs_unique:
        id_split[id] = []

    for i in range(len(beat_IDs)):
        id_split[beat_IDs[i]].append(i)

    # Take a sample of the indexes 
    samples = []
    for id in id_split:
        id_len = len(id_split[id])
        if id_len < num_samples: # Duplicate if there are not enough in the original ID
            id_split[id] = id_split[id] * int(np.ceil(num_samples/id_len))
        samp = np.random.choice(id_split[id], num_samples, replace=False)
        samples.extend(samp)

    # Keep only the sampled indexes
    beats_samp = [beats[i] for i in samples]
    beat_IDs_samp = [beat_IDs[i] for i in samples]

    return beats_samp, beat_IDs_samp

def get_scalogram_from_beat(beat: np.ndarray,
                            widths: np.ndarray,
                            max_length: Optional[int] = None):
    
    """Get scalogram for a single beat"""
    
    scalogram = cwt_single_beat(beat=beat, widths=widths)

    return pad_scalograms(scalogram=scalogram, max_length=max_length)[0]

def get_scalograms_from_signal(signal:np.ndarray, 
                               annotation: wfdb.Annotation,
                               widths: np.ndarray,
                               processes:int = 2,
                               max_length: Optional[int] = None) -> Tuple[List[np.ndarray], List[int]]:
    
    """Get scalogram for all the beats in a filtered signal"""
    
    signal = normalize_signal(signal=signal)
    beats, beat_IDs = segment_signal_relative(signal=signal, annotation=annotation)
    scalograms = cwt_parallel(beats=beats, widths=widths, processes=processes)

    return pad_scalograms(scalograms=scalograms, max_length=max_length), beat_IDs

class ArrhythmiaDatabase(Dataset):

    def __init__(self, 
                 path:str = "data/db.npz") -> None:
        super().__init__()

        npzfile = np.load(path, mmap_mode='r')
        scalograms = npzfile["scalograms"]
        labels = npzfile["labels"]
        self.n = len(labels)

        self.scalograms = th.tensor(scalograms, dtype=th.double).unsqueeze(1)
        self.labels = th.tensor(labels, dtype=th.long)

    def __len__(self) -> int:
        return self.n 

    def __getitem__(self, 
                    idx: int) -> Tuple[th.Tensor, th.Tensor]:
        return self.scalograms[idx], self.labels[idx]   
