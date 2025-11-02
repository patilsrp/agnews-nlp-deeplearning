import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
import re
from typing import Dict, List, Tuple, Optional


class AGNewsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer=None, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'text': text,
                'label': torch.tensor(label, dtype=torch.long)
            }


class DataProcessor:
    def __init__(self):
        self.label_mapping = {
            1: 0,  # World -> 0
            2: 1,  # Sports -> 1
            3: 2,  # Business -> 2
            4: 3   # Science/Technology -> 3
        }
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        self.class_names = ['World', 'Sports', 'Business', 'Science/Technology']
        
    def clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        train_df.columns = ['class', 'title', 'description']
        test_df.columns = ['class', 'title', 'description']
        
        train_df['text'] = train_df['title'] + ' ' + train_df['description']
        test_df['text'] = test_df['title'] + ' ' + test_df['description']
        
        train_df['text'] = train_df['text'].apply(self.clean_text)
        test_df['text'] = test_df['text'].apply(self.clean_text)
        
        train_df['label'] = train_df['class'].map(self.label_mapping)
        test_df['label'] = test_df['class'].map(self.label_mapping)
        
        return train_df, test_df
    
    def create_data_splits(self, train_df: pd.DataFrame, val_size: float = 0.1, random_state: int = 42):
        train_texts = train_df['text'].tolist()
        train_labels = train_df['label'].tolist()
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, 
            test_size=val_size, 
            random_state=random_state, 
            stratify=train_labels
        )
        
        return train_texts, val_texts, train_labels, val_labels
    
    def create_dataloaders(self, 
                          train_texts: List[str], 
                          val_texts: List[str], 
                          test_texts: List[str],
                          train_labels: List[int], 
                          val_labels: List[int], 
                          test_labels: List[int],
                          tokenizer=None,
                          batch_size: int = 32,
                          max_length: int = 128) -> Dict[str, DataLoader]:
        
        train_dataset = AGNewsDataset(train_texts, train_labels, tokenizer, max_length)
        val_dataset = AGNewsDataset(val_texts, val_labels, tokenizer, max_length)
        test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer, max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }


def get_distilbert_tokenizer():
    return DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def load_ag_news_data(data_dir: str = 'data'):
    processor = DataProcessor()
    
    train_path = f"{data_dir}/train.csv"
    test_path = f"{data_dir}/test.csv"
    
    train_df, test_df = processor.load_data(train_path, test_path)
    
    train_texts, val_texts, train_labels, val_labels = processor.create_data_splits(train_df)
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    return {
        'train_texts': train_texts,
        'val_texts': val_texts,
        'test_texts': test_texts,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels,
        'processor': processor
    }