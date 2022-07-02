import argparse
import os

import data
import models
import trainer

import torch
import torch.nn as nn


def train(epochs: int, bactch_size: int, learning_rate: float):
    # read in train_data
    train_dl, valid_dl = data.get_data(bactch_size)
    model = models.Model([784, 100, 100, 10])
    trainer.train(
        model,
        nn.CrossEntropyLoss(),
        learning_rate,
        epochs,
        train_dl,
        valid_dl,
    )
    torch.save(model.state_dict(), os.environ["SM_MODEL_DIR"] + "/model.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args, _ = parser.parse_known_args()
    train(args.epochs, args.batch_size, args.learning_rate)
