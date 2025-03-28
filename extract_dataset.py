import pandas as pd
import numpy as np
import torch

from torcheeg.io.eeg_signal import EEGSignalIO

## Path to dir with data (remember the last '/')
path = "C:/Users/ahmm9/Desktop/v1/"

## Establish connection to datafile
IO = EEGSignalIO(io_path=str(path), io_mode='lmdb')

## Read metadata dataframe
metadata = pd.read_csv(path + 'sample_metadata.tsv', sep='\t')

idxs = np.arange(len(metadata))

eeg = torch.FloatTensor(np.array([IO.read_eeg(str(i)) for i in idxs]))
print(eeg.shape)

torch.save(eeg, "Pipeline/Datasets/emotion_data.pt")
torch.save(torch.tensor(idxs), "Pipeline/Datasets/emotion_idxs.pt")