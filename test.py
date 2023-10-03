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
    model.load_state_dict(th.load('data/models/pace31_2023-09-30 21_20_55.534873.pth', map_location=device)['model'])
    model.eval()
    criterion = nn.CrossEntropyLoss()

    stats = {
        'trial_loss': [],
        'trial_true_label': [],
        'trial_predicted_label': [],
        'sensitivity': {},
        'predicted_probability': [],
        'trial_index': []
    }

    np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
    losses = []
    for i, data in enumerate(dt, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        outputs = model(inputs.unsqueeze(0).to(device))
        loss = criterion(outputs, labels.unsqueeze(0).to(device=device))


        outputs = th.softmax(outputs, dim=1).squeeze().detach().numpy()
        
        # print statistics
        print(f'loss: {loss.item()} true label: {labels} predicted label: {outputs.argmax()} predicted probability: {outputs}')
        losses.append(loss.item())

        stats['trial_loss'].append(loss.item())
        stats['trial_true_label'].append(labels.item())
        stats['trial_predicted_label'].append(outputs.argmax())
        stats['predicted_probability'].append(outputs.tolist())
        stats['trial_index'].append(i)

    print('Finished Testing')

    stats['average_accuracy'] = \
        sum(np.array(stats['trial_predicted_label']) == np.array(stats['trial_true_label']))/len(stats['trial_true_label'])

    for label in np.unique(stats['trial_true_label']):
        mask = stats['trial_true_label'] == label
        stats['sensitivity'][label] = \
            sum(np.array(stats['trial_predicted_label'])[mask] == label)/sum(mask)

    th.save(stats, "data/stats_2023-09-30 21_20_55.534873.pth")

if __name__ == '__main__':
    main()
    