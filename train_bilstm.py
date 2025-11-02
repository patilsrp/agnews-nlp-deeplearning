import torch
import argparse
from src.data import load_ag_news_data, DataProcessor
from src.models import BiLSTMTokenizer, create_bilstm_model
from src.training import Trainer
import os


def main():
    parser = argparse.ArgumentParser(description='Train BiLSTM model for AG News classification')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--embedding_dim', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--save_model', type=str, default='models/bilstm_model.pth', help='Model save path')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data...")
    data = load_ag_news_data(args.data_dir)
    processor = data['processor']
    
    print("Building vocabulary...")
    tokenizer = BiLSTMTokenizer(max_length=args.max_length)
    tokenizer.build_vocab(data['train_texts'])
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    print("Creating data loaders...")
    dataloaders = processor.create_dataloaders(
        data['train_texts'], data['val_texts'], data['test_texts'],
        data['train_labels'], data['val_labels'], data['test_labels'],
        tokenizer=None,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    class BiLSTMDataLoader:
        def __init__(self, texts, labels, tokenizer, batch_size, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.batch_size = batch_size
            self.max_length = max_length
            
        def __iter__(self):
            for i in range(0, len(self.texts), self.batch_size):
                batch_texts = self.texts[i:i+self.batch_size]
                batch_labels = self.labels[i:i+self.batch_size]
                
                batch_encoded = self.tokenizer.batch_encode(batch_texts)
                
                yield {
                    'input_ids': batch_encoded['input_ids'],
                    'attention_mask': batch_encoded['attention_mask'],
                    'label': torch.tensor(batch_labels, dtype=torch.long)
                }
                
        def __len__(self):
            return (len(self.texts) + self.batch_size - 1) // self.batch_size
    
    train_loader = BiLSTMDataLoader(
        data['train_texts'], data['train_labels'], tokenizer, args.batch_size, args.max_length
    )
    val_loader = BiLSTMDataLoader(
        data['val_texts'], data['val_labels'], tokenizer, args.batch_size, args.max_length
    )
    test_loader = BiLSTMDataLoader(
        data['test_texts'], data['test_labels'], tokenizer, args.batch_size, args.max_length
    )
    
    print("Creating model...")
    model = create_bilstm_model(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=4,
        dropout=args.dropout
    )
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("Starting training...")
    trainer = Trainer(model, device, model_type='bilstm')
    
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        early_stopping_patience=5
    )
    
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'model_config': {
            'vocab_size': tokenizer.vocab_size,
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'num_classes': 4,
            'dropout': args.dropout
        },
        'training_results': training_results,
        'test_metrics': test_metrics
    }, args.save_model)
    
    print(f"Model saved to {args.save_model}")
    
    trainer.plot_training_history(save_path='assets/screenshots/bilstm_training_curves.png')
    trainer.plot_confusion_matrix(
        test_metrics['labels'], 
        test_metrics['predictions'],
        processor.class_names,
        save_path='assets/screenshots/bilstm_confusion_matrix.png'
    )


if __name__ == "__main__":
    main()