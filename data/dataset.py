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
    sequence_length: int = 13
    learning_rate: float = 0.001
    num_features: int = 4
    num_epochs: int = 100

class ActivitySequenceDataset(Dataset):
    """
    A PyTorch Dataset for handling sequences of activities with multiple features.
    
    Attributes:
        sequences: List of concatenated feature tensors
        labels: Classification labels
        wt_labels: Requested amount labels
        lengths: Sequence lengths
    """
    
    def __init__(
        self,
        sequences: List[Tuple[np.ndarray, ...]],
        labels: Union[List[int], np.ndarray],
        wt_labels: Union[List[int], np.ndarray],
        lengths: Union[List[int], np.ndarray],
        case_ids: Union[List[str], np.ndarray]
    ):
        # Process all sequences at once using list comprehension
        feature_types = [
            (torch.long, 0), # activities
            (torch.long, 1), # loan goal
            (torch.long, 2), # lf_transitions
            (torch.long, 3), # applicationtype
            (torch.long, 4), # resources
            (torch.float32, 5), # requestedamounts
            (torch.float32, 6), # offeredamounts
            (torch.float32, 7), # monthlycosts
            (torch.float32, 8), # hour of the day
            # (torch.float32, 8), # elapsed days
            # (torch.float32, 9), # creditscore
        ]
        
        # Convert features to tensors efficiently
        features = []
        for dtype, idx in feature_types:
            features.append([torch.tensor(seq[idx], dtype=dtype) for seq in sequences])
            
        # Convert labels and lengths to tensors
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.wt_labels = torch.tensor(wt_labels, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.case_ids = case_ids
        
        # Pre-compute concatenated sequences
        self.sequences = [
            torch.cat(feature_tuple, dim=0)
            for feature_tuple in zip(*features)
        ]
        
        # Optional: Convert sequences to a single tensor for faster batch processing
        # self.sequences = torch.stack(self.sequences)
        
    def __len__(self) -> int:
        """Returns the number of sequences in the dataset."""
        return len(self.sequences)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of (sequence, label, wt_label, length) for the given index.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple containing the sequence and its associated labels and length
        """
        return (
            self.sequences[idx],
            self.labels[idx],
            self.wt_labels[idx],
            self.lengths[idx],
            self.case_ids[idx]
        )
    
from torch.utils.data import Dataset
import torch
import numpy as np
from typing import List, Tuple, Union
from dataclasses import dataclass

class NeSyDataset(Dataset):
    def __init__(self, sequences, labels):
        
        self.sequences = []
        self.labels = torch.tensor(labels, dtype=torch.float)
        for seq in sequences:
            for i in range(len(seq)):
                if i in [8, 21]:
                    seq[i] = torch.tensor(seq[i], dtype=torch.long)
                else:
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
    
class BPI12Dataset(Dataset):
    def __init__(self, sequences, labels):
        
        self.sequences = []
        self.labels = torch.tensor(labels, dtype=torch.float)
        for seq in sequences:
            for i in range(len(seq)):
                if i in [0, 1]:
                    seq[i] = torch.tensor(seq[i], dtype=torch.long)
                else:
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
    
class TrafficDataset(Dataset):
    def __init__(self, sequences, labels):
        
        self.sequences = []
        self.labels = torch.tensor(labels, dtype=torch.float)
        for seq in sequences:
            for i in range(len(seq)):
                if i in [0, 1, 2, 3, 4, 5, 7, 8]:
                    seq[i] = torch.tensor(seq[i], dtype=torch.long)
                else:
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
    
class B17Dataset(Dataset):
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