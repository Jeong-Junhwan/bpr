import torch
from torch import nn
from torch.nn import functional as F


class MatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users + 1, embedding_dim)
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim)

        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        # (Batch) * 2
        user_vector = self.user_embedding(u)
        item_vector = self.item_embedding(i)

        return torch.sum(user_vector * item_vector, dim=1)


class BPR(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.MF = MatrixFactorization(n_users, n_items, embedding_dim)
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch, 3)

        u = x[:, 0]
        i = x[:, 1]
        j = x[:, 2]

        xui = self.MF(u, i)
        xuj = self.MF(u, j)

        xuij = xui - xuj
        output = self.log_sigmoid(xuij)
        # to maximize BPR-OPT
        return -torch.mean(output)
