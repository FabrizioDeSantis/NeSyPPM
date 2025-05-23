import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration class for LSTM model parameters"""
    hidden_size: int
    num_layers: int
    dropout_rate: float = 0.1
    sequence_length: int = 13
    sequence_length: int = 20 # bpi17
    #sequence_length: int = 40 # bpi12
    #sequence_length: int = 13 # sepsis
    #sequence_length: int = 20 # traffic_fines
    learning_rate: float = 0.001
    num_features: int = 4
    num_epochs: int = 100

class LSTMModel(nn.Module):
    def __init__(self, vocab_sizes: List[int], config: ModelConfig, num_classes: int, feature_names: List[str]):
        super(LSTMModel, self).__init__()
        self.config = config
        self.feature_names = feature_names
        self.num_classes = num_classes
        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(vocab_size + 1, config.hidden_size, padding_idx=0)
            for feature, vocab_size in vocab_sizes.items()
        })
        # lstm_input_size = (config.hidden_size * len(self.embeddings)) + (len(feature_names) - len(self.embeddings) - 1)  # +2 for numerical features
        # lstm_input_size = (config.hidden_size * len(self.embeddings)) + (len(feature_names) - len(self.embeddings) - 2) # bpi12/bpi17
        lstm_input_size = (config.hidden_size * len(self.embeddings)) + (len(feature_names) - len(self.embeddings)) # traffic fines
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
        numerical_features = []
        # x = x[...,:364]
        # Process categorical features
        for name in self.embeddings.keys():
            index = self.feature_names.index(name)
            index = index * seq_len
            end_idx = index + seq_len
            feature_data = x[:, index:end_idx].long()
            embeddings_list.append(self.embeddings[name](feature_data))
            
        #Process numerical features
        # SEPSIS
        for name in ['InfectionSuspected', 'DiagnosticBlood', 'DisfuncOrg', 'SIRSCritTachypnea', 'Hypotensie', 'SIRSCritHeartRate', 'Infusion', 'DiagnosticArtAstrup', 'Age', 'DiagnosticIC', 'DiagnosticSputum', 'DiagnosticLiquor', 'DiagnosticOther', 'SIRSCriteria2OrMore', 'DiagnosticXthorax', 'SIRSCritTemperature', 'DiagnosticUrinaryCulture', 'SIRSCritLeucos', 'Oligurie', 'DiagnosticLacticAcid', 'Hypoxie', 'DiagnosticUrinarySediment', 'DiagnosticECG', 'Leucocytes', 'CRP', 'LacticAcid']:
            index = self.feature_names.index(name)
            index = index * seq_len
            end_idx = index + seq_len
            feature_data = x[:, index:end_idx]
            numerical_features.append(feature_data)
        # BPI12
        # index = self.feature_names.index("case:AMOUNT_REQ")
        # index = index * seq_len
        # end_idx = index + seq_len
        # feature_data = x[:, index:end_idx]
        # numerical_features.append(feature_data)
        # traffic
        # for name in ["expense", "amount", "paymentAmount"]:
        #     index = self.feature_names.index(name)
        #     index = index * seq_len
        #     end_idx = index + seq_len
        #     feature_data = x[:, index:end_idx]
        #     numerical_features.append(feature_data)
        # bpi17
        # for name in ["CreditScore", "MonthlyCost", "OfferedAmount", "case:RequestedAmount", "FirstWithdrawalAmount"]:
        #     index = self.feature_names.index(name)
        #     index = index * seq_len
        #     end_idx = index + seq_len
        #     feature_data = x[:, index:end_idx]
        #     numerical_features.append(feature_data)

        numerical_features = torch.stack(numerical_features, dim=2)
        output = torch.cat(embeddings_list + [numerical_features], dim=2)
        # Concatenate all features
        return output

    def forward(self, x):
        cat = self._get_embeddings(x)
        
        output, (hidden, _) = self.lstm(cat)
        last_hidden = hidden[-1]  # Shape: (batch_size, hidden_dim)
        lengths = (x[:, :self.config.sequence_length] != 0).sum(1)  # Mask padding
        last_output = output[torch.arange(output.size(0)), lengths - 1]
        # attention = self.attention_combine(torch.tanh(self.attention_weight(output)))
        # attention_weights = F.softmax(attention, dim=1)
        # context_vector = torch.sum(output * attention_weights, dim=1)
        # return self.sigmoid(self.fc(context_vector))
        out = self.fc(last_output)
        return self.sigmoid(out)
        #return out

class LSTMModelNext(nn.Module):
    def __init__(self, vocab_sizes: List[int], config: ModelConfig, num_classes: int, feature_names: List[str]):
        super(LSTMModelNext, self).__init__()
        self.config = config
        self.feature_names = feature_names
        self.num_classes = num_classes
        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(vocab_size + 1, config.hidden_size, padding_idx=0)
            for feature, vocab_size in vocab_sizes.items()
        })
        lstm_input_size = (config.hidden_size * len(self.embeddings)) + (len(feature_names) - len(self.embeddings))
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
        numerical_features = []
        # Process categorical features
        for name in self.embeddings.keys():
            index = self.feature_names.index(name)
            index = index * seq_len
            end_idx = index + seq_len
            feature_data = x[:, index:end_idx].long()
            embeddings_list.append(self.embeddings[name](feature_data))
            
        #Process numerical features
        # sepsis
        for name in ['InfectionSuspected', 'DiagnosticBlood', 'DisfuncOrg', 'SIRSCritTachypnea', 'Hypotensie', 'SIRSCritHeartRate', 'Infusion', 'DiagnosticArtAstrup', 'Age', 'DiagnosticIC', 'DiagnosticSputum', 'DiagnosticLiquor', 'DiagnosticOther', 'SIRSCriteria2OrMore', 'DiagnosticXthorax', 'SIRSCritTemperature', 'DiagnosticUrinaryCulture', 'SIRSCritLeucos', 'Oligurie', 'DiagnosticLacticAcid', 'Hypoxie', 'DiagnosticUrinarySediment', 'DiagnosticECG', 'Leucocytes', 'CRP', 'LacticAcid']:
            index = self.feature_names.index(name)
            index = index * seq_len
            end_idx = index + seq_len
            feature_data = x[:, index:end_idx]
            numerical_features.append(feature_data)
        # index = self.feature_names.index("case:AMOUNT_REQ")
        # index = index * seq_len
        # end_idx = index + seq_len
        # feature_data = x[:, index:end_idx]
        # numerical_features.append(feature_data)
        # traffic
        # for name in ["expense", "amount", "paymentAmount"]:
        #     index = self.feature_names.index(name)
        #     index = index * seq_len
        #     end_idx = index + seq_len
        #     feature_data = x[:, index:end_idx]
        #     numerical_features.append(feature_data)
        # # bpi17
        # for name in ["CreditScore", "MonthlyCost", "OfferedAmount", "case:RequestedAmount", "FirstWithdrawalAmount"]:
        #     index = self.feature_names.index(name)
        #     index = index * seq_len
        #     end_idx = index + seq_len
        #     feature_data = x[:, index:end_idx]
        #     numerical_features.append(feature_data)

        numerical_features = torch.stack(numerical_features, dim=2)
        output = torch.cat(embeddings_list + [numerical_features], dim=2)
        # Concatenate all features
        return output

    def forward(self, x):
        cat = self._get_embeddings(x)
        
        output, (hidden, _) = self.lstm(cat)
        last_hidden = hidden[-1]  # Shape: (batch_size, hidden_dim)
        lengths = (x[:, :self.config.sequence_length] != 0).sum(1)  # Mask padding
        last_output = output[torch.arange(output.size(0)), lengths - 1]
        # attention = self.attention_combine(torch.tanh(self.attention_weight(output)))
        # attention_weights = F.softmax(attention, dim=1)
        # context_vector = torch.sum(output * attention_weights, dim=1)
        # return self.sigmoid(self.fc(context_vector))
        out = self.fc(last_output)
        return self.sigmoid(out)
    
class LogitsToPredicate(nn.Module):
    def __init__(self, logits_model):
        super(LogitsToPredicate, self).__init__()
        self.logits_model = logits_model
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, l):
        logits = self.logits_model(x)
        probs = self.softmax(logits)
        out = torch.sum(probs * l, dim=1)
        return out

class LSTMModelA(nn.Module):
    def __init__(self, vocab_sizes: List[int], config: ModelConfig, num_classes: int, feature_names: List[str]):
        super(LSTMModelA, self).__init__()
        self.config = config
        self.feature_names = feature_names
        self.num_classes = num_classes
        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(vocab_size + 1, config.hidden_size, padding_idx=0)
            for feature, vocab_size in vocab_sizes.items()
        })
        lstm_input_size = (config.hidden_size * len(self.embeddings)) + (len(feature_names) - len(self.embeddings)) + 3  # +2 for numerical features
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
        numerical_features = []
        # Process categorical features
        for name in self.embeddings.keys():
            index = self.feature_names.index(name)
            index = index * seq_len
            end_idx = index + seq_len
            feature_data = x[:, index:end_idx].long()
            embeddings_list.append(self.embeddings[name](feature_data))
            
        # Process numerical features
        # sepsis
        for name in ['InfectionSuspected', 'DiagnosticBlood', 'DisfuncOrg', 'SIRSCritTachypnea', 'Hypotensie', 'SIRSCritHeartRate', 'Infusion', 'DiagnosticArtAstrup', 'Age', 'DiagnosticIC', 'DiagnosticSputum', 'DiagnosticLiquor', 'DiagnosticOther', 'SIRSCriteria2OrMore', 'DiagnosticXthorax', 'SIRSCritTemperature', 'DiagnosticUrinaryCulture', 'SIRSCritLeucos', 'Oligurie', 'DiagnosticLacticAcid', 'Hypoxie', 'DiagnosticUrinarySediment', 'DiagnosticECG', 'Leucocytes', 'CRP', 'LacticAcid', "rule_2"]:
            index = self.feature_names.index(name)
            index = index * seq_len
            end_idx = index + seq_len
            feature_data = x[:, index:end_idx]
            numerical_features.append(feature_data)
        # for name in ["expense", "amount", "paymentAmount"]:
        #     index = self.feature_names.index(name)
        #     index = index * seq_len
        #     end_idx = index + seq_len
        #     feature_data = x[:, index:end_idx]
        #     numerical_features.append(feature_data)
        # numerical_features.append(x[:, 110:120])
        # numerical_features.append(x[:, 120:130])
        # numerical_features.append(x[:, 130:140])
        # # bpi17
        # for name in ["CreditScore", "MonthlyCost", "OfferedAmount", "case:RequestedAmount", "FirstWithdrawalAmount"]:
        #     index = self.feature_names.index(name)
        #     index = index * seq_len
        #     end_idx = index + seq_len
        #     feature_data = x[:, index:end_idx]
        #     numerical_features.append(feature_data)
        # numerical_features.append(x[:, 240:260])
        # numerical_features.append(x[:, 260:280])
        # numerical_features.append(x[:, 280:300])

        numerical_features = torch.stack(numerical_features, dim=2)
        output = torch.cat(embeddings_list + [numerical_features], dim=2)
        # Concatenate all features
        return output

    def forward(self, x):
        cat = self._get_embeddings(x)
        
        output, (hidden, _) = self.lstm(cat)
        last_hidden = hidden[-1]  # Shape: (batch_size, hidden_dim)
        lengths = (x[:, :self.config.sequence_length] != 0).sum(1)  # Mask padding
        last_output = output[torch.arange(output.size(0)), lengths - 1]
        out = self.fc(last_output)
        return self.sigmoid(out)
        #return out