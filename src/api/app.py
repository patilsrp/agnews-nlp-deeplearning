from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json
import io
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import create_distilbert_model, create_bilstm_model, BiLSTMTokenizer
from src.data import get_distilbert_tokenizer, DataProcessor
from src.utils.explainer import TextExplainer


app = FastAPI(title="AG News Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str
    model_type: str = "distilbert"


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    attention_weights: Optional[List[float]] = None


class ModelManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.distilbert_model = None
        self.bilstm_model = None
        self.distilbert_tokenizer = None
        self.bilstm_tokenizer = None
        self.processor = DataProcessor()
        self.explainer = None
        
    def load_distilbert_model(self, model_path: str = "models/distilbert_model.pth"):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model_config = checkpoint['model_config']
            
            self.distilbert_model = create_distilbert_model(**model_config)
            self.distilbert_model.load_state_dict(checkpoint['model_state_dict'])
            self.distilbert_model.to(self.device)
            self.distilbert_model.eval()
            
            self.distilbert_tokenizer = get_distilbert_tokenizer()
            print("DistilBERT model loaded successfully")
            
        except Exception as e:
            print(f"Error loading DistilBERT model: {e}")
            
    def load_bilstm_model(self, model_path: str = "models/bilstm_model.pth"):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model_config = checkpoint['model_config']
            
            self.bilstm_model = create_bilstm_model(**model_config)
            self.bilstm_model.load_state_dict(checkpoint['model_state_dict'])
            self.bilstm_model.to(self.device)
            self.bilstm_model.eval()
            
            self.bilstm_tokenizer = checkpoint['tokenizer']
            print("BiLSTM model loaded successfully")
            
        except Exception as e:
            print(f"Error loading BiLSTM model: {e}")
    
    def predict_distilbert(self, text: str) -> Dict:
        if self.distilbert_model is None or self.distilbert_tokenizer is None:
            raise HTTPException(status_code=500, detail="DistilBERT model not loaded")
        
        cleaned_text = self.processor.clean_text(text)
        
        encoding = self.distilbert_tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.distilbert_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=1)
            
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            prob_dict = {
                self.processor.class_names[i]: probabilities[0][i].item()
                for i in range(len(self.processor.class_names))
            }
            
            attention_weights = None
            if 'attention_weights' in outputs and outputs['attention_weights'] is not None:
                attention_weights = outputs['attention_weights'][0].cpu().numpy().tolist()
        
        return {
            'prediction': self.processor.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': prob_dict,
            'attention_weights': attention_weights
        }
    
    def predict_bilstm(self, text: str) -> Dict:
        if self.bilstm_model is None or self.bilstm_tokenizer is None:
            raise HTTPException(status_code=500, detail="BiLSTM model not loaded")
        
        cleaned_text = self.processor.clean_text(text)
        
        encoded = self.bilstm_tokenizer.encode(cleaned_text)
        input_ids = encoded['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = encoded['attention_mask'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.bilstm_model(input_ids=input_ids, mask=attention_mask)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=1)
            
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            prob_dict = {
                self.processor.class_names[i]: probabilities[0][i].item()
                for i in range(len(self.processor.class_names))
            }
            
            attention_weights = None
            if 'attention_weights' in outputs:
                attention_weights = outputs['attention_weights'][0].cpu().numpy().tolist()
        
        return {
            'prediction': self.processor.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': prob_dict,
            'attention_weights': attention_weights
        }


model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    if os.path.exists("models/distilbert_model.pth"):
        model_manager.load_distilbert_model()
    if os.path.exists("models/bilstm_model.pth"):
        model_manager.load_bilstm_model()


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AG News Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            textarea { width: 100%; height: 150px; margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            select, button { padding: 10px; margin: 5px; border: 1px solid #ddd; border-radius: 5px; }
            button { background-color: #007bff; color: white; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            .result { margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }
            .prob-bar { background-color: #e9ecef; height: 20px; border-radius: 10px; margin: 5px 0; }
            .prob-fill { height: 100%; border-radius: 10px; background-color: #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 AG News Topic Classifier</h1>
            <p>Enter a news headline or article text to classify it into one of four categories: World, Sports, Business, or Science/Technology.</p>
            
            <textarea id="textInput" placeholder="Enter news text here...">Apple announces new iPhone with revolutionary camera technology and improved battery life.</textarea>
            
            <div>
                <select id="modelSelect">
                    <option value="distilbert">DistilBERT</option>
                    <option value="bilstm">BiLSTM + Attention</option>
                </select>
                <button onclick="classify()">Classify Text</button>
                <button onclick="clearText()">Clear</button>
            </div>
            
            <div id="result" class="result" style="display: none;">
                <h3>Prediction Result:</h3>
                <div id="predictionContent"></div>
            </div>
        </div>

        <script>
            async function classify() {
                const text = document.getElementById('textInput').value;
                const modelType = document.getElementById('modelSelect').value;
                
                if (!text.trim()) {
                    alert('Please enter some text to classify.');
                    return;
                }
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text, model_type: modelType })
                    });
                    
                    const result = await response.json();
                    displayResult(result);
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
            
            function displayResult(result) {
                const resultDiv = document.getElementById('result');
                const contentDiv = document.getElementById('predictionContent');
                
                let html = `
                    <p><strong>Predicted Category:</strong> ${result.prediction}</p>
                    <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>
                    <h4>Class Probabilities:</h4>
                `;
                
                for (const [className, prob] of Object.entries(result.probabilities)) {
                    const percentage = (prob * 100).toFixed(2);
                    html += `
                        <div>
                            <span>${className}: ${percentage}%</span>
                            <div class="prob-bar">
                                <div class="prob-fill" style="width: ${percentage}%"></div>
                            </div>
                        </div>
                    `;
                }
                
                contentDiv.innerHTML = html;
                resultDiv.style.display = 'block';
            }
            
            function clearText() {
                document.getElementById('textInput').value = '';
                document.getElementById('result').style.display = 'none';
            }
        </script>
    </body>
    </html>
    """


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    try:
        if input_data.model_type == "distilbert":
            result = model_manager.predict_distilbert(input_data.text)
        elif input_data.model_type == "bilstm":
            result = model_manager.predict_bilstm(input_data.text)
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        return PredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...), model_type: str = "distilbert"):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if 'text' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'text' column")
        
        results = []
        for text in df['text']:
            if model_type == "distilbert":
                result = model_manager.predict_distilbert(str(text))
            else:
                result = model_manager.predict_bilstm(str(text))
            results.append(result)
        
        return {"predictions": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/status")
async def model_status():
    return {
        "distilbert_loaded": model_manager.distilbert_model is not None,
        "bilstm_loaded": model_manager.bilstm_model is not None,
        "device": str(model_manager.device)
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)