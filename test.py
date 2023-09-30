import numpy as np
import torch as th
import torch.nn as nn
from pace import db
import pace.model as md

"""
Author: Alice Cheng, Josue N Rivera
Date: 9/19/2023
"""
def main():
    dt = db.ArrhythmiaDatabase("data/db_31_test.npz")
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    scalo1, _ = dt[0]
    
    model = md.Pace(scalo1.size(2)).to(device=device)
    model.load_state_dict(th.load('data/models/pace31_2023-09-30 12_39_29.540086.pth', map_location=device)['model'])
    model.eval()
    criterion = nn.CrossEntropyLoss()

    np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
    losses = []
    for _, data in enumerate(dt, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        outputs = model(inputs.unsqueeze(0).to(device))
        loss = criterion(outputs, labels.unsqueeze(0).to(device=device))


        outputs = th.softmax(outputs, dim=1).squeeze().detach().numpy()
        
        # print statistics
        print(f'loss: {loss.item()} true label: {labels} predicted label: {outputs.argmax()} predicted probability: {outputs}')
        losses.append(loss.item())

    print('Finished Testing')

if __name__ == '__main__':
    main()
    