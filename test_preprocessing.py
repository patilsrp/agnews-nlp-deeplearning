#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

from data.data_loader import load_ag_news_data, get_distilbert_tokenizer

def test_preprocessing():
    print("🔍 Testing data preprocessing pipeline...")
    
    # Load and preprocess data
    data = load_ag_news_data()
    
    print(f"\n📊 Data Statistics:")
    print(f"Train samples: {len(data['train_texts'])}")
    print(f"Validation samples: {len(data['val_texts'])}")
    print(f"Test samples: {len(data['test_texts'])}")
    
    # Display sample data
    print(f"\n📝 Sample processed text:")
    print(f"Text: {data['train_texts'][0][:100]}...")
    print(f"Label: {data['train_labels'][0]} ({data['processor'].class_names[data['train_labels'][0]]})")
    
    # Test tokenizer
    print(f"\n🔤 Testing tokenizer...")
    tokenizer = get_distilbert_tokenizer()
    sample_text = data['train_texts'][0]
    
    encoding = tokenizer(
        sample_text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    print(f"Input IDs shape: {encoding['input_ids'].shape}")
    print(f"Attention mask shape: {encoding['attention_mask'].shape}")
    
    # Test data loader creation
    print(f"\n🔄 Testing data loaders...")
    dataloaders = data['processor'].create_dataloaders(
        data['train_texts'][:100],  # Small sample for testing
        data['val_texts'][:50],
        data['test_texts'][:50],
        data['train_labels'][:100],
        data['val_labels'][:50],
        data['test_labels'][:50],
        tokenizer=tokenizer,
        batch_size=8
    )
    
    # Test one batch
    train_batch = next(iter(dataloaders['train']))
    print(f"Batch input_ids shape: {train_batch['input_ids'].shape}")
    print(f"Batch labels shape: {train_batch['label'].shape}")
    
    print(f"\n✅ Data preprocessing pipeline working correctly!")

if __name__ == "__main__":
    test_preprocessing()