import torch
import argparse
from src.data import load_ag_news_data, DataProcessor, get_distilbert_tokenizer
from src.models import create_distilbert_model
from src.training import Trainer
import os


def main():
    parser = argparse.ArgumentParser(description='Train DistilBERT model for AG News classification')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder layers')
    parser.add_argument('--use_attention', action='store_true', help='Use attention mechanism')
    parser.add_argument('--save_model', type=str, default='models/distilbert_model.pth', help='Model save path')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data...")
    data = load_ag_news_data(args.data_dir)
    processor = data['processor']
    
    print("Loading tokenizer...")
    tokenizer = get_distilbert_tokenizer()
    
    print("Creating data loaders...")
    dataloaders = processor.create_dataloaders(
        data['train_texts'], data['val_texts'], data['test_texts'],
        data['train_labels'], data['val_labels'], data['test_labels'],
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    print("Creating model...")
    model = create_distilbert_model(
        num_classes=4,
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder,
        use_attention=args.use_attention
    )
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if args.freeze_encoder:
        print("Encoder layers are frozen")
    
    print("Starting training...")
    trainer = Trainer(model, device, model_type='distilbert')
    
    training_results = trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        early_stopping_patience=3
    )
    
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(dataloaders['test'])
    
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_classes': 4,
            'dropout': args.dropout,
            'freeze_encoder': args.freeze_encoder,
            'use_attention': args.use_attention
        },
        'training_results': training_results,
        'test_metrics': test_metrics
    }, args.save_model)
    
    print(f"Model saved to {args.save_model}")
    
    trainer.plot_training_history(save_path='assets/screenshots/distilbert_training_curves.png')
    trainer.plot_confusion_matrix(
        test_metrics['labels'], 
        test_metrics['predictions'],
        processor.class_names,
        save_path='assets/screenshots/distilbert_confusion_matrix.png'
    )


if __name__ == "__main__":
    main()