from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import Model


class LightningModule(pl.LightningModule):
    def __init__(self, model: Model, loss_fn: nn.Module, lr: float):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr

    def common_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor]):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        accuracy = (logits.argmax(dim=-1) == y).float().mean()
        return loss, accuracy

    def training_step(
        self, batch: Tuple[torch.FloatTensor, torch.LongTensor], batch_idx: int
    ) -> torch.FloatTensor:
        loss, accuracy = self.common_step(batch)
        self.log("train_accuracy", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], batch_idx: int):
        _, accuracy = self.common_step(batch)
        self.log("valid_accuracy", accuracy, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), self.lr)


def train(
    model: Model,
    loss_fn: nn.Module,
    learning_rate: float,
    epochs: int,
    train_dl: DataLoader,
    valid_dl: DataLoader,
):
    lightning_module = LightningModule(model, loss_fn, learning_rate)
    trainer = pl.Trainer(accelerator="auto", max_epochs=epochs)
    trainer.fit(lightning_module, train_dl, valid_dl)
