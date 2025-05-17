import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from torch.nn.utils.rnn import pack_padded_sequence

from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration class for LSTM model parameters"""
    hidden_size: int
    num_layers: int
    dropout_rate: float = 0.1
    sequence_length: int = 25
    learning_rate: float = 0.001
    num_features: int = 4

class LSTMModel(nn.Module):
    def __init__(self, vocab_sizes: List[int], config: ModelConfig, num_classes: int):
        super(LSTMModel, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.embeddings = nn.ModuleDict({
            'activity': nn.Embedding(vocab_sizes[0] + 1, config.hidden_size, padding_idx=0),
            'goal': nn.Embedding(vocab_sizes[4] + 1, config.hidden_size, padding_idx=0),
            'lf': nn.Embedding(vocab_sizes[1] + 1, config.hidden_size, padding_idx=0),
            'apptype': nn.Embedding(vocab_sizes[2] + 1, config.hidden_size, padding_idx=0),
            'resource': nn.Embedding(vocab_sizes[3] + 1, config.hidden_size, padding_idx=0)
        })
        lstm_input_size = (config.hidden_size * len(self.embeddings)) + self.config.num_features  # +2 for numerical features
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout_rate
        )
        self.fc = nn.Linear(config.hidden_size, self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = self.config.sequence_length
        embeddings_list = []
        
        # Process categorical features
        start_idx = 0
        for name in self.embeddings.keys():
            end_idx = start_idx + seq_len
            print("*****************")
            print(x.shape)
            print("*****************")
            feature_data = x[:, start_idx:end_idx].long()
            embeddings_list.append(self.embeddings[name](feature_data))
            start_idx = end_idx
            
        # Process numerical features
        num_features_start = seq_len * len(self.embeddings)
        numerical_features = x[:, num_features_start:num_features_start + self.config.num_features * seq_len]
        print(numerical_features.shape)
        numerical_features = numerical_features.view(x.size(0), seq_len, self.config.num_features)
        print(numerical_features.shape)
        
        # Concatenate all features
        return torch.cat(embeddings_list + [numerical_features], dim=2)

    def forward(self, x):
        cat = self._get_embeddings(x)
        print(cat[0])
        print(cat[0].shape)
        print(cat[0][0])
        print(cat.shape)
        out, _ = self.lstm(cat)
        lengths = (x[:, :25] != 0).sum(1)  # Mask padding
        last_output = out[torch.arange(out.size(0)), lengths - 1]
        return self.sigmoid(self.fc(last_output))
    
class LogitsToPredicateWSoftmax(torch.nn.Module):
    def __init__(self, logits_model):
        super(LogitsToPredicateWSoftmax, self).__init__()
        self.logits_model = logits_model
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x, l):
        logits = self.logits_model(x)
        probs = self.softmax(logits)
        out = torch.sum(probs * l, dim=1)
        return out
    
class LogitsToPredicateWSigmoid(torch.nn.Module):
    def __init__(self, logits_model):
        super(LogitsToPredicateWSigmoid, self).__init__()
        self.logits_model = logits_model
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        logits = self.logits_model(x)
        probs = self.sigmoid(logits)
        return probs