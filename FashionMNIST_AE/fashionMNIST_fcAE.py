#!/usr/bin/env python
# coding: utf-8

from utils import mnist_reader

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt
import time


def main():
    # X is images, y is labels
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    X_train = np.float32(X_train)
    x_std = np.std(X_train)
    x_mean = np.mean(X_train)

    model = Autoencoder()
    model.to(device)
    model_param = model.parameters()
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model_param), lr=0.0001)
    criterion = nn.MSELoss()
    dataset = torch.from_numpy(X_train)
    dataset_norm = (dataset-torch.mean(dataset))/torch.std(dataset)

    # Training variables
    epochs = 100
    batch_size = 1024

    start_time = time.time()
    best_loss = training(model, dataset_norm, optimizer, criterion,  Epochs=epochs, batch_size=batch_size, device=device)
    time_taken = time.time() - start_time

    with open('result.txt', 'a') as file:
        file.write(time.strftime("%H:%M:%S", time.gmtime()))
        file.write(f"\nBest Loss: {best_loss:.6f} found in {time_taken:.6f}ms.")
        file.write(f"\nDetails:\n\t- epochs: {epochs}\n\t- batch_size: {batch_size}\n\t- device: {device}")
        file.write("\n\n")

    with torch.no_grad():
        result = model(dataset.to(device))
        result = result.detach().cpu().numpy() * x_std + x_mean

    print(len(result))
    plt.imshow(result[5].reshape([28, 28]))
    # plt.show()


class Autoencoder(nn.Module):
    # Modify dimensions here.
    INPUT_DIM = 784
    TRANS2_DIM = INPUT_DIM // 2
    TRANS3_DIM = TRANS2_DIM // 2
    LATENT_DIM = 8

    def __init__(self):
        super().__init__()

        self.enc1 = nn.Linear(in_features=self.INPUT_DIM, out_features=self.TRANS2_DIM)
        self.enc2 = nn.Linear(in_features=self.TRANS2_DIM, out_features=self.TRANS3_DIM)
        self.enc3 = nn.Linear(in_features=self.TRANS3_DIM, out_features=self.LATENT_DIM)

        self.dec1 = nn.Linear(in_features=self.LATENT_DIM, out_features=self.TRANS3_DIM)
        self.dec2 = nn.Linear(in_features=self.TRANS3_DIM, out_features=self.TRANS2_DIM)
        self.dec3 = nn.Linear(in_features=self.TRANS2_DIM, out_features=self.INPUT_DIM)

        self.encodings = [
            self.enc1,
            self.enc2,
            self.enc3,
        ]

        self.decodings = [
            self.dec1,
            self.dec2,
            self.dec3,
        ]

    def encode(self, x):
        for e in self.encodings:
            x = e(x)
            x = nn.LeakyReLU(0.5)(x)
        return x
    
    def decode(self, x):
        for d in self.decodings:
            x = d(x)
            x = nn.LeakyReLU(0.5)(x)
        return x

    def forward(self, x):
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon


def shuffle_data(data: Tensor) -> Tensor:
    """
    Shuffles the rows of a PyTorch tensor along the first dimension.

    Args:
    - data: A PyTorch tensor of shape (N, ...), where N is the number of samples.

    Returns:
    - new_data: A shuffled PyTorch tensor of the same shape as the input tensor.
    """
    size = data.shape[0]
    index = torch.randperm(size)
    new_data = data[index]
    return new_data


def training(model: Autoencoder,
             dataset: Tensor,
             optimizer: Adam,
             criterion: nn.MSELoss,
             Epochs: int,
             batch_size: int = 1000,
             device: str = "cpu"):

    """
    Trains the given autoencoder model on the given dataset using the specified optimizer and loss criterion for a specified number of epochs.

    Args:
    - model (Autoencoder): The autoencoder model to train.
    - dataset (nn.Tensor): The dataset to train the autoencoder on.
    - optimizer (nn.Adam): The optimizer used to adjust the model's parameters during training.
    - criterion (nn.MSELoss): The loss criterion used to evaluate the quality of the autoencoder's output compared to the original data.
    - Epochs (int): The number of epochs to train the autoencoder for.
    - batch_size (int, optional): The size of the batches to use during training. Defaults to 1000.
    - device (str, optional): The device to use during training (e.g. 'cpu', 'cuda'). Defaults to 'cpu'.
    - early_stop_patience (int, optional): Number of epochs to wait without improvement before stopping
    - validation_split: (float, optional): Fraction of the data to use for validation.

    Returns:
    - train_loss (list of float): The list of training losses for each epoch of training.
    """

    train_loss = []
    best_loss = 1e10
    dataset = shuffle_data(dataset).to(device)

    data_size = dataset.shape[0]
    batches = data_size//batch_size

    for epoch in range(Epochs):
        running_loss = 0.0

        for i in range(batches):
            start = i*batch_size
            end = min(data_size, start+batch_size)

            data = dataset[start:end]   

            optimizer.zero_grad()
            # print(data)
            outputs = model(data).to(device)
            loss = criterion(outputs, data)
            # print(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / batches

        train_loss.append(loss)

        if epoch % 20 == 0:
            print('Epoch {} of {}, Train Loss: {:.6f}'.format(epoch+1, Epochs, running_loss))

        if (loss < best_loss) and (epoch % 20 == 0):
            torch.save(model.state_dict(), './results/latent_{}_best_parameters.pt'.format(model.LATENT_DIM))
            #torch.save(model.state_dict(), './results/AE/v2_{}/model/AE_loge_allplanes_latent{}_best_parameters.pt'.format(timestep, latent_dim))
            best_loss = loss
            print('best loss: ', best_loss)


    return best_loss


main()
