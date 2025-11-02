import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_weights = self.attention(lstm_output)
        attention_weights = attention_weights.squeeze(-1)
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_weights, dim=1)
        
        weighted_output = torch.bmm(attention_weights.unsqueeze(1), lstm_output)
        weighted_output = weighted_output.squeeze(1)
        
        return weighted_output, attention_weights


class BiLSTMWithAttention(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 300,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_classes: int = 4,
                 dropout: float = 0.3,
                 pretrained_embeddings: Optional[torch.Tensor] = None):
        super(BiLSTMWithAttention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = AttentionLayer(hidden_dim * 2)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        lstm_output, (hidden, cell) = self.lstm(embedded)
        
        attended_output, attention_weights = self.attention(lstm_output, mask)
        
        attended_output = self.dropout(attended_output)
        logits = self.fc(attended_output)
        
        return {
            'logits': logits,
            'attention_weights': attention_weights,
            'lstm_output': lstm_output
        }


class BiLSTMTokenizer:
    def __init__(self, max_vocab_size: int = 50000, max_length: int = 128):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        
    def build_vocab(self, texts: list):
        word_freq = {}
        for text in texts:
            words = text.split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        for word, freq in sorted_words:
            if self.vocab_size >= self.max_vocab_size:
                break
            if word not in self.word_to_idx:
                self.word_to_idx[word] = self.vocab_size
                self.idx_to_word[self.vocab_size] = word
                self.vocab_size += 1
    
    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        words = text.split()[:self.max_length]
        input_ids = [self.word_to_idx.get(word, 1) for word in words]
        
        attention_mask = [1] * len(input_ids)
        
        while len(input_ids) < self.max_length:
            input_ids.append(0)
            attention_mask.append(0)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def batch_encode(self, texts: list) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_attention_mask = []
        
        for text in texts:
            encoded = self.encode(text)
            batch_input_ids.append(encoded['input_ids'])
            batch_attention_mask.append(encoded['attention_mask'])
        
        return {
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_mask)
        }


def load_glove_embeddings(glove_path: str, word_to_idx: Dict[str, int], embedding_dim: int = 300) -> torch.Tensor:
    embeddings = torch.randn(len(word_to_idx), embedding_dim)
    
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word_to_idx:
                vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32)
                embeddings[word_to_idx[word]] = vector
    
    embeddings[0] = torch.zeros(embedding_dim)
    
    return embeddings


def create_bilstm_model(vocab_size: int, 
                       embedding_dim: int = 300,
                       hidden_dim: int = 256,
                       num_layers: int = 2,
                       num_classes: int = 4,
                       dropout: float = 0.3,
                       pretrained_embeddings: Optional[torch.Tensor] = None) -> BiLSTMWithAttention:
    
    model = BiLSTMWithAttention(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        pretrained_embeddings=pretrained_embeddings
    )
    
    return model