from utils import network_model
from utils import dataloader
import torch
import argparse
import os

ROOT_DIR = os.path.abspath("./")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataPath', type=str, help='Folder which include output-PlanePoint.npy', nargs=1)
    parser.add_argument('--epoch', type=int,
                        default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch_size')
    args = parser.parse_args()
    ####### parameters ##########
    DATA_PATH = args.dataPath[0]
    epochs = args.epoch
    batch_size = args.batch_size

    ######### model and dataloader #########
    mdn = network_model.MDN(2, 50, 50, 3, 3)
    dataset = dataloader.RoadPlaneDataset(DATA_PATH)

    optimizer = torch.optim.Adam(mdn.parameters(), lr=0.01) # create an optimizer
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True) # create a data loader

    for epoch in range(epochs):
        epoch_loss = 0 # track the epoch loss
        for x, y in loader: # iterate over the data loader
            pi, mu, sigma = mdn(x) # forward pass
            loss = mdn.mdn_loss(pi, mu, sigma, y) # compute the loss
            optimizer.zero_grad() # zero the gradients
            loss.backward() # backward pass
            optimizer.step() # update the parameters
            epoch_loss += loss.item() # update the epoch loss
        print(f"Epoch {epoch+1}, Loss {epoch_loss/len(loader)}") # print the epoch loss

    save_path = os.path.join(OUTPUT_DIR, 'mdn.pt')
    torch.save(mdn, save_path)

    print('Save trained MDN model to {}'.format(save_path))
