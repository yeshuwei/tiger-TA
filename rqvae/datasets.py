import pandas as pd
import torch
import torch.utils.data as data
import numpy as np


class EmbDataset(data.Dataset):

    def __init__(self, data_path, return_index=False):

        self.data_path = data_path
        self.return_index = return_index
        self.embeddings = pd.read_parquet(data_path)['embedding'].values
        self.embeddings = np.stack(self.embeddings, axis=0)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        if self.return_index:
            return index, tensor_emb
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
