import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

OUT_PATH = Path(f"../autoencoder/{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}/")
OUT_PATH.mkdir(parents=True, exist_ok=True)
lr = 0.1
weight_decay = 1
epochs = 1000
ep_log_interval = 100
batch_size = 256
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")


class AE(torch.nn.Module):
    def __init__(self, k: int, n_features: int):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_features, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, k),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(k, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_features),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class CustomDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).to(device)


def auto_encoder_fit_predict(k: int, data: np.ndarray):
    model = AE(k, data.shape[1]).to(device)
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = CustomDataset(data)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
    loss_history = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for (ix, batch) in enumerate(data_loader):
            _, reconstructed = model(batch)
            loss = mse_loss(reconstructed, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().numpy()
        loss_history.append(epoch_loss)
        if epoch % ep_log_interval == 0:
            print("epoch = %4d loss = %0.4f" % (epoch, epoch_loss))
    plt.clf()
    plt.plot(loss_history)
    plt.title("Autoencoder Loss Over Time")
    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"{OUT_PATH}/loss-{k}-{uuid.uuid4()}.png")
    model.eval()
    with torch.no_grad():
        result = model.encoder(torch.Tensor(data).to(device)).cpu().detach().numpy()
    return result
