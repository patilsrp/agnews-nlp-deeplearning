import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class Trainer:
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 model_type: str = 'distilbert'):
        self.model = model
        self.device = device
        self.model_type = model_type
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, 
                   dataloader: DataLoader, 
                   optimizer: optim.Optimizer, 
                   criterion: nn.Module,
                   scheduler: Optional = None) -> Tuple[float, float]:
        
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            if self.model_type == 'distilbert':
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
            else:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, mask=attention_mask)
                logits = outputs['logits']
            
            loss = criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def validate_epoch(self, 
                      dataloader: DataLoader, 
                      criterion: nn.Module) -> Tuple[float, float, Dict]:
        
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Validation")
            
            for batch in progress_bar:
                if self.model_type == 'distilbert':
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs['logits']
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(input_ids=input_ids, mask=attention_mask)
                    logits = outputs['logits']
                
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }
        
        return avg_loss, accuracy, metrics
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 10,
              learning_rate: float = 2e-5,
              weight_decay: float = 0.01,
              warmup_steps: int = 500,
              early_stopping_patience: int = 3) -> Dict:
        
        criterion = nn.CrossEntropyLoss()
        
        if self.model_type == 'distilbert':
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            total_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2, verbose=True
            )
        
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        print(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, criterion, 
                scheduler if self.model_type == 'distilbert' else None
            )
            
            val_loss, val_acc, val_metrics = self.validate_epoch(val_loader, criterion)
            
            if self.model_type == 'bilstm' and scheduler:
                scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            
            if early_stopping(val_loss, self.model):
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'training_time': training_time,
            'final_metrics': val_metrics
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, test_metrics = self.validate_epoch(test_loader, criterion)
        
        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        print(f"Test F1-Score: {test_metrics['f1']:.4f}")
        
        return test_metrics
    
    def plot_training_history(self, save_path: Optional[str] = None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                            class_names: List[str], save_path: Optional[str] = None):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def save_model(model: nn.Module, save_path: str, tokenizer=None):
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_type': type(model).__name__,
        }
    }, save_path)
    
    if tokenizer:
        if hasattr(tokenizer, 'save_pretrained'):
            tokenizer.save_pretrained(save_path.replace('.pth', '_tokenizer'))


def load_model(model: nn.Module, load_path: str):
    checkpoint = torch.load(load_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model