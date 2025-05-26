import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from typing import List, Tuple, Union
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration class for LSTM model parameters"""
    hidden_size: int
    num_layers: int
    dropout_rate: float = 0.1
    sequence_length: int = 13 # sepsis
    # sequence_length: int = 20 # bpi17
    #sequence_length: int = 40 # bpi12
    #sequence_length: int = 13 # sepsis
    #sequence_length: int = 20 # traffic_fines
    learning_rate: float = 0.001
    num_features: int = 4
    num_epochs: int = 100
    
class NeSyDataset(Dataset):
    def __init__(self, sequences, labels):
        
        self.sequences = []
        self.labels = torch.tensor(labels, dtype=torch.float)
        for seq in sequences:
            for i in range(len(seq)):
                seq[i] = torch.tensor(seq[i], dtype=torch.float32)
            self.sequences.append(seq)
        
        self.sequences = [
            torch.cat(seq, dim=0) for seq in self.sequences
        ]

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.labels[idx]
        )