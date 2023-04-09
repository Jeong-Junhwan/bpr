from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


class BPRDataset(Dataset):
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> torch.Tensor:
        # Too many negative items, just randomly select the negative samples.
        user_id, positive, negatives = self.data.iloc[index]
        negative = np.random.choice(negatives)
        return torch.LongTensor([user_id, positive, negative])
