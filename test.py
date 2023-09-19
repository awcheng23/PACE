import numpy as np
import torch as th
import torch.nn as nn
from pace.db import get_patients_beats, cwt_parallel, padd_scalograms
import pace.model as md

"""
Author: Alice Cheng, Josue N Rivera
Date: 9/19/2023
"""
def main():
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    model = md.Pace().to(device=device)
    model.load_state_dict(th.load('data/models/pace31.pth', map_location=device)['model'])
    model.eval()
    criterion = nn.CrossEntropyLoss()

    pat_beats, pat_beat_ID = get_patients_beats(ID=103, dt_path='data/mitdb/')
    scalograms = cwt_parallel(beats=pat_beats, widths=np.arange(1, 31))
    scalograms = np.array(padd_scalograms(scalograms, 966))
    pat_beat_ID = th.tensor(pat_beat_ID)

    losses = []
    for inputs, labels in zip(th.tensor(scalograms), pat_beat_ID):

        outputs = model(inputs.unsqueeze(0).unsqueeze(0).to(device))
        loss = criterion(outputs, labels.unsqueeze(0).to(device=device))

        outputs = th.softmax(outputs, dim=1).squeeze().detach().numpy()
        
        # print statistics
        print(f'loss: {loss.item()} true label: {labels} predicted probability: {outputs}')
        losses.append(loss.item())

    print('Finished Testing')

if __name__ == '__main__':
    main()