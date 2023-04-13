import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from config import *


class CoLADataset(Dataset):
    def __init__(self, path, tokenizer):
        self.df = pd.read_csv(path,
                              sep='\t',
                              names=["x", "label", "y", "sentence"],
                              header=None
                              )[['label', 'sentence']]

        self.data = self.df.to_dict(orient='records')
        self.tokenizer = tokenizer

    def get_dataloader(self):
        return DataLoader(self.data, batch_size=BATCH_SIZE, shuffle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_pt = self.data[idx]

        tokenized_sentence = self.tokenizer.encode_plus(
            data_pt['sentence'],
            return_tensors='pt',
            padding="max_length",
            # max_length=15,
            truncation=True
        )
        label = torch.tensor([data_pt["label"]], dtype=torch.float)

        return tokenized_sentence, label
        # return {
        #     "labels": label,
        #     **tokenized_sentence
        # }