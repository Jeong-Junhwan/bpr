import pytest
from model import MatrixFactorization, BPR
import torch


def test_MatrixFactorization():
    model = MatrixFactorization(n_users=10, n_items=15, embedding_dim=16)
    data = torch.LongTensor([[1, 5], [5, 7], [5, 1], [4, 2]])
    u = data[:, 0]
    i = data[:, 1]

    assert len(model.forward(u, i)) == 4


def test_BPR():
    model = BPR(n_users=10, n_items=15, embedding_dim=16)
    data = torch.LongTensor([[1, 5, 6], [2, 5, 7], [3, 5, 1], [4, 5, 2]])

    loss = model.forward(data)
    loss.backward()
