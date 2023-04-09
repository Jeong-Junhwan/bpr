import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm


class BPRTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_data_loader: DataLoader,
        valid_data_loader: DataLoader,
        device: torch.device,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        self.model = model
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.device = device
        self.epochs = epochs

        self.optimizer = torch.optim.SGD(
            params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def train(self):
        self.model = self.model.to(self.device)
        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)
            self._valid_epoch(epoch)

    def _train_epoch(self, epoch: int):
        self.model.train()
        progress_bar = tqdm(
            self.train_data_loader, desc=f"{epoch}/{self.epochs} Train epoch"
        )

        for batch_idx, data in enumerate(progress_bar, 1):
            data = data.to(self.device)
            loss = self.model(data)  # using own criterion (BPR)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch_idx % 20 == 0:
                progress_bar.set_postfix_str(f"loss: {round(loss.item(), 8)}")

    @torch.no_grad()
    def _valid_epoch(self, epoch: int):
        self.model.eval()

        losses = []
        for batch_idx, data in enumerate(self.valid_data_loader, 1):
            data = data.to(self.device)
            loss = self.model(data)  # using own criterion (BPR)

            losses.append(loss.item())

        mean_loss = np.mean(losses)

        print(f"{epoch}/{self.epochs} average valid loss: {round(mean_loss, 8)}")
        return mean_loss

    def _early_stopping(self):
        pass
