import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import os
from src.data import load_ag_news_data, get_distilbert_tokenizer
from src.models import create_distilbert_model, create_bilstm_model
from src.utils import explain_prediction


def load_model_and_results(model_path: str, model_type: str):
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if model_type == 'distilbert':
        model = create_distilbert_model(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        tokenizer = get_distilbert_tokenizer()
    else:
        model = create_bilstm_model(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        tokenizer = checkpoint['tokenizer']
    
    training_results = checkpoint.get('training_results', {})
    test_metrics = checkpoint.get('test_metrics', {})
    
    return model, tokenizer, training_results, test_metrics


def plot_training_curves(training_results: dict, model_name: str, save_path: str = None):
    if not training_results:
        print(f"No training results found for {model_name}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(training_results['train_losses']) + 1)
    
    ax1.plot(epochs, training_results['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, training_results['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title(f'{model_name} - Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, training_results['train_accuracies'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, training_results['val_accuracies'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title(f'{model_name} - Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, model_name: str, save_path: str = None):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Count'})
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def compare_models(distilbert_results: dict, bilstm_results: dict, save_path: str = None):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    distilbert_values = [distilbert_results.get(metric, 0) for metric in metrics]
    bilstm_values = [bilstm_results.get(metric, 0) for metric in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, distilbert_values, width, label='DistilBERT', color='skyblue')
    bars2 = ax.bar(x + width/2, bilstm_values, width, label='BiLSTM + Attention', color='lightcoral')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_class_performance(test_metrics: dict, class_names: list, model_name: str, save_path: str = None):
    if 'labels' not in test_metrics or 'predictions' not in test_metrics:
        print(f"Insufficient data for class analysis in {model_name}")
        return
    
    report = classification_report(
        test_metrics['labels'], 
        test_metrics['predictions'], 
        target_names=class_names,
        output_dict=True
    )
    
    class_metrics = pd.DataFrame({
        class_name: {
            'Precision': report[class_name]['precision'],
            'Recall': report[class_name]['recall'],
            'F1-Score': report[class_name]['f1-score'],
            'Support': report[class_name]['support']
        }
        for class_name in class_names
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    class_metrics.iloc[:3].T.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title(f'{model_name} - Per-Class Performance Metrics')
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Score')
    ax1.legend(title='Metrics')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    support_data = class_metrics.iloc[3]
    ax2.pie(support_data, labels=class_names, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'{model_name} - Test Set Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return class_metrics


def create_sample_explanations(model, tokenizer, class_names: list, model_name: str, device):
    sample_texts = [
        "Apple reports record quarterly earnings with strong iPhone sales in international markets",
        "Manchester United defeats Liverpool 3-1 in Premier League championship match",
        "Scientists discover new exoplanet with potential for supporting life forms",
        "Global stock markets rally as inflation concerns ease worldwide"
    ]
    
    print(f"\n{model_name} - Sample Predictions and Explanations:")
    print("=" * 60)
    
    for i, text in enumerate(sample_texts):
        print(f"\nSample {i+1}: {text[:80]}...")
        
        try:
            explanation = explain_prediction(
                model, tokenizer, text, class_names, device,
                save_dir=f"assets/screenshots/{model_name.lower()}_explanation_{i+1}"
            )
            
            print(f"Predicted: {explanation['summary']['predicted_class']}")
            print(f"Confidence: {explanation['summary']['confidence']:.3f}")
            print(f"Top positive words: {explanation['summary']['top_positive_words'][:3]}")
            
        except Exception as e:
            print(f"Error generating explanation: {e}")


def main():
    parser = argparse.ArgumentParser(description='Visualize and analyze model results')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--distilbert_model', type=str, default='models/distilbert_model.pth', help='DistilBERT model path')
    parser.add_argument('--bilstm_model', type=str, default='models/bilstm_model.pth', help='BiLSTM model path')
    parser.add_argument('--output_dir', type=str, default='assets/screenshots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    data = load_ag_news_data(args.data_dir)
    class_names = data['processor'].class_names
    
    distilbert_results = None
    bilstm_results = None
    
    if os.path.exists(args.distilbert_model):
        print("Loading DistilBERT model and results...")
        distilbert_model, distilbert_tokenizer, distilbert_training, distilbert_test = load_model_and_results(
            args.distilbert_model, 'distilbert'
        )
        distilbert_model.to(device)
        distilbert_results = distilbert_test
        
        plot_training_curves(
            distilbert_training, 'DistilBERT',
            save_path=f"{args.output_dir}/distilbert_training_curves.png"
        )
        
        if 'labels' in distilbert_test and 'predictions' in distilbert_test:
            plot_confusion_matrix(
                distilbert_test['labels'], distilbert_test['predictions'], 
                class_names, 'DistilBERT',
                save_path=f"{args.output_dir}/distilbert_confusion_matrix.png"
            )
            
            analyze_class_performance(
                distilbert_test, class_names, 'DistilBERT',
                save_path=f"{args.output_dir}/distilbert_class_performance.png"
            )
        
        create_sample_explanations(distilbert_model, distilbert_tokenizer, class_names, 'DistilBERT', device)
    
    if os.path.exists(args.bilstm_model):
        print("Loading BiLSTM model and results...")
        bilstm_model, bilstm_tokenizer, bilstm_training, bilstm_test = load_model_and_results(
            args.bilstm_model, 'bilstm'
        )
        bilstm_model.to(device)
        bilstm_results = bilstm_test
        
        plot_training_curves(
            bilstm_training, 'BiLSTM + Attention',
            save_path=f"{args.output_dir}/bilstm_training_curves.png"
        )
        
        if 'labels' in bilstm_test and 'predictions' in bilstm_test:
            plot_confusion_matrix(
                bilstm_test['labels'], bilstm_test['predictions'], 
                class_names, 'BiLSTM + Attention',
                save_path=f"{args.output_dir}/bilstm_confusion_matrix.png"
            )
            
            analyze_class_performance(
                bilstm_test, class_names, 'BiLSTM + Attention',
                save_path=f"{args.output_dir}/bilstm_class_performance.png"
            )
        
        create_sample_explanations(bilstm_model, bilstm_tokenizer, class_names, 'BiLSTM', device)
    
    if distilbert_results and bilstm_results:
        print("Comparing models...")
        compare_models(
            distilbert_results, bilstm_results,
            save_path=f"{args.output_dir}/model_comparison.png"
        )
    
    print(f"All visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()