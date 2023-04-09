import pytest
from dataset import BPRDataset
from preprocess import load_train_data
import numpy as np
import torch

from torch.utils.data import DataLoader


def test_BPRDataset():
    inter_data, data_info = load_train_data()
    train_dataset = BPRDataset(inter_data)

    assert len(train_dataset) == data_info["ratings"]
    assert len(train_dataset[0]) == 3
    assert len(train_dataset[5]) == 3
    assert len(train_dataset[100]) == 3
    assert isinstance(train_dataset[100], torch.LongTensor)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
    )

    duplicate_check = set()
    for i, x in enumerate(train_data_loader):
        if i > 500:
            break

        duplicate_check.add(x)

        assert isinstance(x, torch.Tensor)
        assert x.shape == (1, 3)

    assert len(duplicate_check) == i
