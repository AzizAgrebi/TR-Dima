import os
import librosa
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
import torchvision
import torch.nn.functional as F

from data.Create_Floorplans import PointCloudProcessor
from data.CustomDataset import CustomDataset
from model.CustomLoss import WeightedL1Loss
from model.unet import build_unet


# Création des ground truths
paths = [d for d in os.listdir('datasets/') if os.path.isdir(os.path.join('datasets/', d)) and d != ".ipynb_checkpoints"]
print(paths)

Creation_data = PointCloudProcessor(1280, 720, 639.764, 639.764, 641.364, 356.711)
Creation_data.process_point_clouds(paths)


# Définition des ensembles de données
train_dataset = CustomDataset('train.csv', "")
val_dataset = CustomDataset('val.csv', "")

# Définition des loaders de données
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# Entrainement du modèle avec loss l1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_unet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = torch.nn.L1Loss().to(device)

T = []
V = []
nb_epoch = 100

for epoch in range(nb_epoch):
    model.train() # Mode entraînement
    print('\nEpoch: {:.6f}\n'.format(epoch))
    # Boucle d'entraînement
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad() # Initialisation du gradient
        data = data.to(device)
        target = target.to(device)
        output = model(data) # Prédiction
        loss = loss_fn(output, target) # Calcul de la perte
        loss.backward() # Calcul du gradient
        optimizer.step() # Mise à jour des poids
        train_loss += loss.detach().item()
    train_loss /= len(train_loader.dataset)
    T.append(train_loss)
    print('\nTrain set: Average loss: {:.6f}\n'.format(train_loss))
   

    # Boucle d'évaluation
    model.eval() # Mode évaluation
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            val_loss += loss_fn(output, target).detach().item() # Calcul de la perte

    val_loss /= len(val_loader.dataset)
    V.append(val_loss)
    print('\nVal set: Average loss: {:.6f}\n'.format(val_loss))

torch.save(model.state_dict(), 'unet_model_l1.pt')
plt.plot(T, label='Train')
plt.plot(V, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('learning_curves.png')


# Entrainement du modèle avec weighted loss l1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_unet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = WeightedL1Loss.to(device)

T = []
V = []
nb_epoch = 100

for epoch in range(nb_epoch):
    model.train() # Mode entraînement
    print('\nEpoch: {:.6f}\n'.format(epoch))
    # Boucle d'entraînement
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad() # Initialisation du gradient
        data = data.to(device)
        target = target.to(device)
        output = model(data) # Prédiction
        loss = loss_fn(output, target) # Calcul de la perte
        loss.backward() # Calcul du gradient
        optimizer.step() # Mise à jour des poids
        train_loss += loss.detach().item()
    train_loss /= len(train_loader.dataset)
    T.append(train_loss)
    print('\nTrain set: Average loss: {:.6f}\n'.format(train_loss))
   

    # Boucle d'évaluation
    model.eval() # Mode évaluation
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            val_loss += loss_fn(output, target).detach().item() # Calcul de la perte

    val_loss /= len(val_loader.dataset)
    V.append(val_loss)
    print('\nVal set: Average loss: {:.6f}\n'.format(val_loss))

torch.save(model.state_dict(), 'unet_model_weighted_l1.pt')
plt.plot(T, label='Train')
plt.plot(V, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('learning_curves.png')
