from datetime import datetime
import torch as th
import torch.nn as nn
import torch.optim as optim
import pace.db as db
import pace.model as md

"""
Author: Alice Cheng, Josue N Rivera
Date: 9/1/2023
"""
def main():
    dt = db.ArrhythmiaDatabase("data/db_31_train.npz")
    trainloader = th.utils.data.DataLoader(dt, batch_size=10, shuffle=True)
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    scalo1, _ = dt[0]

    model = md.Pace(scalo1.size(2)).to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 

    losses = []
    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device=device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 250 == 249:    # print every 250 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 250:.3f}')
                running_loss = 0.0
                losses.append(running_loss / 250)

    print('Finished Training')

    checkpoint = { 
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'losses': losses}
    th.save(checkpoint, f"data/models/pace31_{str(datetime.now()).replace(':','_')}.pth")
if __name__ == '__main__':
    main()
    