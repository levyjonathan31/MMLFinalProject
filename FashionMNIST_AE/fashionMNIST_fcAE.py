#!/usr/bin/env python
# coding: utf-8

from utils import mnist_reader

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

from autoencoder import Autoencoder

import numpy as np
import matplotlib.pyplot as plt
import time


def main():
    # Run training
    do_train = False
    # X is images, y is labels
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    # Normalize Data
    X_std = np.std(X_train)
    X_mean = np.mean(X_train)
    X_train = (X_train - X_mean) / X_std

    X_test = (X_test - X_mean) / X_std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running with", device)

    X_train = np.float32(X_train)

    # Setting up model parameters
    model = Autoencoder()
    model.to(device)
    model_param = model.parameters()
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model_param), lr=0.0005)
    criterion = nn.MSELoss()
    dataset = torch.from_numpy(X_train).to(device)
    dataset_norm = (dataset - torch.mean(dataset)) / torch.std(dataset).to(device)

    # Training variables
    EPOCHS = 1000
    BATCH_SIZE = 2048

    # Training
    start_time = time.time()
    if do_train:
        best_loss = training(model, dataset_norm, optimizer, criterion, Epochs=EPOCHS, batch_size=BATCH_SIZE, device=device)
    train_time_taken = time.time() - start_time

    # Test and Training output
    state_dict = torch.load('results/latent_39_best_parameters.pt')
    model.load_state_dict(state_dict)

    # Calculating Result
    with torch.no_grad():
        result = model(dataset.to(device))
        result = result.detach().cpu().numpy() * X_std + X_mean
        dataset_cpu = dataset.detach().cpu().numpy() * X_std + X_mean

    # Linear Least Squares
    ls_time_start = time.time()

    # Tensor declarations
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Least Squares on the autoencoded data
    # Training data
    ae_training_data: Tensor
    ae_test_data: Tensor
    with torch.no_grad():
        ae_training_data = model.forward(X_train_tensor.to(device))
        ae_training_latent = model.encode(X_train_tensor.to(device)).detach().cpu().numpy()
    latent_training = least_squares(model.to(device), ae_training_data).detach().cpu().numpy()
    # Test data
    with torch.no_grad():
        ae_test_data = model.forward(X_test_tensor.to(device))
        ae_test_latent = model.encode(X_test_tensor.to(device)).detach().cpu().numpy()
    latent_test = least_squares(model.to(device), ae_test_data).detach().cpu().numpy()
    print("Latent Space Difference 2-norm Training: ", np.linalg.norm(latent_training - ae_training_latent))
    print("Latent Space Difference 2-norm Testing: ", np.linalg.norm(latent_test - ae_test_latent))

    # Least squares on the original data
    # Training data
    latent_training = least_squares(model.to(device), X_train_tensor.to(device))
    # Test data
    latent_test = least_squares(model.to(device), X_test_tensor.to(device))
    ls_time_taken = time.time() - ls_time_start
    top_layer_training = model.decode(latent_training.to(device)).detach().cpu().numpy()
    top_layer_test = model.decode(latent_test.to(device)).detach().cpu().numpy()
    print("Top Layer Difference 2-norm Training: ", np.linalg.norm(top_layer_training - X_train))
    print("Top Layer Difference 2-norm Test: ", np.linalg.norm(top_layer_test - X_test))

    # Calculate the compression ratio and output the diagnostics
    if do_train:
        comp_ratio = model.compute_compression_ratio(dataset_norm)
        write_diagnostics(model, best_loss, train_time_taken, ls_time_taken, comp_ratio, EPOCHS, BATCH_SIZE, device)

    # Display stuff
    n = 4
    fig, axs = plt.subplots(n, n * 2, figsize=(8, 8))
    for i in range(n):
        for j in range(n):
            idx = np.random.randint(60000)
            if idx < len(result):
                axs[i, j].imshow(dataset_cpu[idx].reshape([28, 28]), cmap='gray')  # Note: dataset_cpu is pulled from "dataset", which itself is pulled from TRAINING data.
                axs[i, j].axis('off')
                axs[i, j + n].imshow(result[idx].reshape([28, 28]), cmap='gray')
                axs[i, j + n].axis('off')
    plt.show()


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

    Returns:
    - train_loss (list of float): The list of training losses for each epoch of training.
    """

    train_loss = []
    best_loss = 1e10
    dataset = shuffle_data(dataset)

    data_size = dataset.shape[0]
    batches = data_size // batch_size

    for epoch in range(Epochs):
        running_loss = 0.0

        for i in range(batches):
            start = i * batch_size
            end = min(data_size, start + batch_size)

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
            print('Epoch {} of {}, Train Loss: {:.6f}'.format(epoch + 1, Epochs, running_loss))

        if (loss < best_loss) and (epoch % 20 == 0):
            torch.save(model.state_dict(), './results/latent_{}_best_parameters.pt'.format(model.LATENT_DIM))
            # torch.save(model.state_dict(), './results/AE/v2_{}/model/AE_loge_allplanes_latent{}_best_parameters.pt'.format(timestep, latent_dim))
            best_loss = loss
            print('best loss: ', best_loss)

    return best_loss


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


def least_squares(model: Autoencoder, dataset: Tensor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    b = dataset.T.to(device)

    for dec in reversed(model.decodings):
        A = torch.tensor(dec.weight, dtype=torch.float32, device=device)
        next_input, _, _, _ = torch.linalg.lstsq(A, b)
        next_input = torch.where(next_input < 0, next_input / model.LR_FACTOR, next_input)
        b = next_input

    return b.T



def nrmse(x, x_recon):
    return np.sqrt(np.mean((x - x_recon) ** 2)) / (np.max(x) - np.min(x))

def write_diagnostics(model, best_loss, time_taken, ls_time_taken, comp_ratio, epochs: int, batch_size: int, device: str):

    with open('result.txt', 'r+') as file:
        # Read the contents and store them
        contents = file.read()

        # Write what we need at the top of the file
        file.seek(0)
        file.write(time.strftime("%H:%M:%S", time.gmtime()))
        file.write(f"\nBest Loss: {best_loss:.6f} found in {time_taken:.6f}s.")
        file.write(f"\nLeast squares took {ls_time_taken:.6f}s.")
        file.write(f"\nCompression ratio: {comp_ratio:.2f}x")
        file.write("\nDetails:")
        file.write(f"\n\t- epochs: {epochs}")
        file.write(f"\n\t- batch_size: {batch_size}")
        file.write(f"\n\t- device: {device}")
        file.write(f"\n\t- latent size: {model.LATENT_DIM}")
        file.write("\n\n")

        # Push the stuff that was previously there below the new contents
        file.write(contents)


main()
