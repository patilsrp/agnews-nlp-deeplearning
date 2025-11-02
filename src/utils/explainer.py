import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import lime
from lime.lime_text import LimeTextExplainer
import shap
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


class TextExplainer:
    def __init__(self, model, tokenizer, class_names: List[str], device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.device = device
        self.lime_explainer = LimeTextExplainer(class_names=class_names)
        
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for text in texts:
                if hasattr(self.tokenizer, 'encode'):
                    encoded = self.tokenizer.encode(text)
                    input_ids = encoded['input_ids'].unsqueeze(0).to(self.device)
                    attention_mask = encoded['attention_mask'].unsqueeze(0).to(self.device)
                    
                    outputs = self.model(input_ids=input_ids, mask=attention_mask)
                else:
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=128,
                        return_tensors='pt'
                    )
                    input_ids = encoding['input_ids'].to(self.device)
                    attention_mask = encoding['attention_mask'].to(self.device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=1)
                probabilities.append(probs.cpu().numpy()[0])
        
        return np.array(probabilities)
    
    def explain_lime(self, text: str, num_features: int = 10, num_samples: int = 1000) -> Dict:
        explanation = self.lime_explainer.explain_instance(
            text, 
            self.predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )
        
        explanation_dict = {
            'text': text,
            'prediction': explanation.predict_proba.argmax(),
            'prediction_label': self.class_names[explanation.predict_proba.argmax()],
            'confidence': explanation.predict_proba.max(),
            'feature_importance': explanation.as_list(),
            'explanation': explanation
        }
        
        return explanation_dict
    
    def get_attention_weights(self, text: str) -> Tuple[List[str], List[float]]:
        self.model.eval()
        
        with torch.no_grad():
            if hasattr(self.tokenizer, 'encode'):
                encoded = self.tokenizer.encode(text)
                input_ids = encoded['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = encoded['attention_mask'].unsqueeze(0).to(self.device)
                
                outputs = self.model(input_ids=input_ids, mask=attention_mask)
                tokens = text.split()[:len(encoded['input_ids'])]
            else:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                
                tokens = [token for token in tokens if token not in ['[PAD]', '[CLS]', '[SEP]']]
            
            if 'attention_weights' in outputs:
                attention_weights = outputs['attention_weights'][0].cpu().numpy()
                
                valid_length = min(len(tokens), len(attention_weights))
                tokens = tokens[:valid_length]
                attention_weights = attention_weights[:valid_length]
                
                return tokens, attention_weights.tolist()
            else:
                return tokens, [1.0 / len(tokens)] * len(tokens)
    
    def visualize_attention(self, text: str, save_path: Optional[str] = None):
        tokens, attention_weights = self.get_attention_weights(text)
        
        plt.figure(figsize=(15, 8))
        
        colors = plt.cm.Reds(np.array(attention_weights) / max(attention_weights))
        
        bars = plt.bar(range(len(tokens)), attention_weights, color=colors)
        plt.xlabel('Tokens')
        plt.ylabel('Attention Weight')
        plt.title('Attention Weights Visualization')
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        
        for i, (token, weight) in enumerate(zip(tokens, attention_weights)):
            plt.text(i, weight + 0.001, f'{weight:.3f}', 
                    ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return tokens, attention_weights
    
    def visualize_lime_explanation(self, explanation_dict: Dict, save_path: Optional[str] = None):
        feature_importance = explanation_dict['feature_importance']
        
        words = [item[0] for item in feature_importance]
        importance = [item[1] for item in feature_importance]
        
        colors = ['red' if imp < 0 else 'green' for imp in importance]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(words, importance, color=colors, alpha=0.7)
        plt.xlabel('Feature Importance')
        plt.title(f'LIME Explanation for: {explanation_dict["prediction_label"]}')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        for i, (word, imp) in enumerate(zip(words, importance)):
            plt.text(imp + (0.01 if imp >= 0 else -0.01), i, f'{imp:.3f}', 
                    va='center', ha='left' if imp >= 0 else 'right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_word_cloud(self, explanation_dict: Dict, save_path: Optional[str] = None):
        feature_importance = explanation_dict['feature_importance']
        
        word_weights = {}
        for word, importance in feature_importance:
            if importance > 0:
                word_weights[word] = importance
        
        if word_weights:
            wordcloud = WordCloud(
                width=800, height=400, 
                background_color='white',
                max_words=50,
                colormap='viridis'
            ).generate_from_frequencies(word_weights)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Important Words for {explanation_dict["prediction_label"]} Classification')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_comprehensive_explanation(self, text: str, save_dir: Optional[str] = None) -> Dict:
        print("Generating LIME explanation...")
        lime_explanation = self.explain_lime(text)
        
        print("Extracting attention weights...")
        tokens, attention_weights = self.get_attention_weights(text)
        
        print("Creating visualizations...")
        
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            self.visualize_lime_explanation(
                lime_explanation, 
                save_path=f"{save_dir}/lime_explanation.png"
            )
            
            self.visualize_attention(
                text, 
                save_path=f"{save_dir}/attention_weights.png"
            )
            
            self.create_word_cloud(
                lime_explanation,
                save_path=f"{save_dir}/word_cloud.png"
            )
        else:
            self.visualize_lime_explanation(lime_explanation)
            self.visualize_attention(text)
            self.create_word_cloud(lime_explanation)
        
        comprehensive_explanation = {
            'input_text': text,
            'lime_explanation': lime_explanation,
            'attention': {
                'tokens': tokens,
                'weights': attention_weights
            },
            'summary': {
                'predicted_class': lime_explanation['prediction_label'],
                'confidence': lime_explanation['confidence'],
                'top_positive_words': [item for item in lime_explanation['feature_importance'] if item[1] > 0][:5],
                'top_negative_words': [item for item in lime_explanation['feature_importance'] if item[1] < 0][:5]
            }
        }
        
        return comprehensive_explanation


def create_explainer(model, tokenizer, class_names: List[str], device: torch.device) -> TextExplainer:
    return TextExplainer(model, tokenizer, class_names, device)


def explain_prediction(model, tokenizer, text: str, class_names: List[str], 
                      device: torch.device, save_dir: Optional[str] = None) -> Dict:
    explainer = create_explainer(model, tokenizer, class_names, device)
    return explainer.generate_comprehensive_explanation(text, save_dir)