import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
from typing import Dict, Optional


class DistilBertForSequenceClassification(nn.Module):
    def __init__(self, 
                 model_name: str = 'distilbert-base-uncased',
                 num_classes: int = 4,
                 dropout: float = 0.3,
                 freeze_encoder: bool = False):
        super(DistilBertForSequenceClassification, self).__init__()
        
        self.num_classes = num_classes
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        
        if freeze_encoder:
            for param in self.distilbert.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, 
                return_dict: bool = True) -> Dict[str, torch.Tensor]:
        
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        pooled_output = outputs.last_hidden_state[:, 0]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if return_dict:
            return {
                'logits': logits,
                'hidden_states': outputs.last_hidden_state,
                'attention_weights': outputs.attentions if hasattr(outputs, 'attentions') else None
            }
        else:
            return logits


class DistilBertClassifierWithAttention(nn.Module):
    def __init__(self, 
                 model_name: str = 'distilbert-base-uncased',
                 num_classes: int = 4,
                 dropout: float = 0.3,
                 freeze_encoder: bool = False):
        super(DistilBertClassifierWithAttention, self).__init__()
        
        self.num_classes = num_classes
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        
        if freeze_encoder:
            for param in self.distilbert.parameters():
                param.requires_grad = False
        
        self.attention_weights = nn.Linear(self.distilbert.config.hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.attention_weights.weight)
        nn.init.constant_(self.attention_weights.bias, 0)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, 
                return_dict: bool = True) -> Dict[str, torch.Tensor]:
        
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        
        attention_scores = self.attention_weights(sequence_output).squeeze(-1)
        
        attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        weighted_output = torch.bmm(attention_probs.unsqueeze(1), sequence_output).squeeze(1)
        
        weighted_output = self.dropout(weighted_output)
        logits = self.classifier(weighted_output)
        
        if return_dict:
            return {
                'logits': logits,
                'hidden_states': sequence_output,
                'attention_weights': attention_probs,
                'cls_output': outputs.last_hidden_state[:, 0]
            }
        else:
            return logits


def create_distilbert_model(model_name: str = 'distilbert-base-uncased',
                           num_classes: int = 4,
                           dropout: float = 0.3,
                           freeze_encoder: bool = False,
                           use_attention: bool = False) -> nn.Module:
    
    if use_attention:
        model = DistilBertClassifierWithAttention(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout,
            freeze_encoder=freeze_encoder
        )
    else:
        model = DistilBertForSequenceClassification(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout,
            freeze_encoder=freeze_encoder
        )
    
    return model


def get_model_size(model: nn.Module) -> Dict[str, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params
    }