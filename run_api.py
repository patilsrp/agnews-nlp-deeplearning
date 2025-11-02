import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
import io
import re
from typing import Dict, List
import uvicorn
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize FastAPI app
app = FastAPI(title="AG News Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class TextInput(BaseModel):
    text: str
    model_type: str = "distilbert"

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]

# Simple data processor
class DataProcessor:
    def __init__(self):
        self.class_names = ['World', 'Sports', 'Business', 'Science/Technology']
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = DataProcessor()
distilbert_model = None
distilbert_tokenizer = None

# Model loading functions
def load_distilbert_model():
    """Load DistilBERT model with saved weights"""
    global distilbert_model, distilbert_tokenizer
    try:
        # Load tokenizer
        distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Load model architecture
        distilbert_model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', 
            num_labels=4
        )
        
        # Load saved weights
        if os.path.exists('models/distilbert_model.pth'):
            state_dict = torch.load('models/distilbert_model.pth', map_location=device)
            distilbert_model.load_state_dict(state_dict)
            distilbert_model.to(device)
            distilbert_model.eval()
            print("✅ DistilBERT model loaded successfully")
            return True
        else:
            print("❌ DistilBERT model file not found at models/distilbert_model.pth")
            return False
            
    except Exception as e:
        print(f"❌ Error loading DistilBERT model: {e}")
        return False

# Prediction functions
def predict_distilbert(text: str) -> Dict:
    """Make prediction using DistilBERT model"""
    if distilbert_model is None or distilbert_tokenizer is None:
        raise HTTPException(status_code=500, detail="DistilBERT model not loaded")
    
    cleaned_text = processor.clean_text(text)
    
    encoding = distilbert_tokenizer(
        cleaned_text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = distilbert_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        prob_dict = {
            processor.class_names[i]: probabilities[0][i].item()
            for i in range(len(processor.class_names))
        }
    
    return {
        'prediction': processor.class_names[predicted_class],
        'confidence': confidence,
        'probabilities': prob_dict
    }

# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Load models when the API starts"""
    print("🔄 Loading models...")
    load_distilbert_model()

# API endpoints
@app.get("/", response_class=HTMLResponse)
async def home():
    """Main web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AG News Classifier</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 900px; 
                margin: auto; 
                background: white; 
                padding: 40px; 
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h1 { 
                color: #333; 
                text-align: center; 
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            textarea { 
                width: 100%; 
                height: 120px; 
                margin: 15px 0; 
                padding: 15px; 
                border: 2px solid #e1e5e9; 
                border-radius: 8px; 
                font-size: 16px;
                resize: vertical;
                transition: border-color 0.3s;
            }
            textarea:focus {
                outline: none;
                border-color: #007bff;
            }
            .controls {
                display: flex;
                gap: 10px;
                margin: 20px 0;
                justify-content: center;
                flex-wrap: wrap;
            }
            button { 
                padding: 12px 24px; 
                border: none; 
                border-radius: 6px; 
                background-color: #007bff; 
                color: white; 
                cursor: pointer; 
                font-size: 16px;
                font-weight: 500;
                transition: all 0.3s;
                min-width: 120px;
            }
            button:hover { 
                background-color: #0056b3; 
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,123,255,0.3);
            }
            button:disabled {
                background-color: #6c757d;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }
            .clear-btn {
                background-color: #6c757d;
            }
            .clear-btn:hover {
                background-color: #545b62;
            }
            .result { 
                margin-top: 30px; 
                padding: 25px; 
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                border-radius: 10px; 
                border-left: 5px solid #007bff;
            }
            .prediction-header {
                font-size: 1.3em;
                font-weight: bold;
                color: #007bff;
                margin-bottom: 15px;
            }
            .confidence {
                font-size: 1.1em;
                margin-bottom: 20px;
                color: #28a745;
                font-weight: 500;
            }
            .prob-container {
                margin-top: 20px;
            }
            .prob-item {
                margin: 10px 0;
            }
            .prob-label {
                display: flex;
                justify-content: space-between;
                margin-bottom: 5px;
                font-weight: 500;
            }
            .prob-bar { 
                background-color: #e9ecef; 
                height: 25px; 
                border-radius: 12px; 
                position: relative;
                overflow: hidden;
            }
            .prob-fill { 
                height: 100%; 
                border-radius: 12px; 
                background: linear-gradient(90deg, #007bff, #0056b3);
                transition: width 0.8s ease-in-out;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                padding-right: 10px;
                color: white;
                font-weight: bold;
                font-size: 12px;
            }
            .loading {
                display: none;
                text-align: center;
                color: #007bff;
                font-weight: 500;
            }
            .examples {
                margin-top: 30px;
                padding: 20px;
                background-color: #f1f3f4;
                border-radius: 8px;
            }
            .example-btn {
                background-color: #28a745;
                font-size: 14px;
                padding: 8px 16px;
                margin: 5px;
                min-width: auto;
            }
            .example-btn:hover {
                background-color: #218838;
            }
            @media (max-width: 768px) {
                .container { padding: 20px; }
                h1 { font-size: 2em; }
                .controls { flex-direction: column; align-items: center; }
                button { width: 100%; max-width: 300px; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 AG News Topic Classifier</h1>
            <p class="subtitle">Powered by DistilBERT • Deep Learning Text Classification</p>
            <p style="text-align: center; color: #666;">
                Enter a news headline or article text to classify it into one of four categories: 
                <strong>World</strong>, <strong>Sports</strong>, <strong>Business</strong>, or <strong>Science/Technology</strong>
            </p>
            
            <textarea id="textInput" placeholder="Enter news text here... (e.g., 'Apple announces new iPhone with revolutionary camera technology')">Apple announces new iPhone with revolutionary camera technology and improved battery life.</textarea>
            
            <div class="controls">
                <button id="classifyBtn" onclick="classify()">🔍 Classify Text</button>
                <button class="clear-btn" onclick="clearText()">🗑️ Clear</button>
            </div>
            
            <div class="loading" id="loading">
                🤖 Analyzing text...
            </div>
            
            <div id="result" class="result" style="display: none;">
                <div class="prediction-header" id="predictionHeader"></div>
                <div class="confidence" id="confidenceText"></div>
                <div class="prob-container">
                    <h4 style="margin-bottom: 15px;">📊 Class Probabilities:</h4>
                    <div id="probabilityBars"></div>
                </div>
            </div>
            
            <div class="examples">
                <h4>💡 Try these examples:</h4>
                <button class="example-btn" onclick="setExample('Tesla stock soars after quarterly earnings beat expectations')">Business</button>
                <button class="example-btn" onclick="setExample('Scientists discover water on Mars using new rover technology')">Science</button>
                <button class="example-btn" onclick="setExample('Manchester United defeats Barcelona 3-1 in Champions League final')">Sports</button>
                <button class="example-btn" onclick="setExample('President announces new trade agreement with European Union')">World</button>
            </div>
        </div>

        <script>
            async function classify() {
                const text = document.getElementById('textInput').value;
                const classifyBtn = document.getElementById('classifyBtn');
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                
                if (!text.trim()) {
                    alert('Please enter some text to classify.');
                    return;
                }
                
                // Show loading state
                classifyBtn.disabled = true;
                classifyBtn.textContent = '🤖 Analyzing...';
                loading.style.display = 'block';
                result.style.display = 'none';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text, model_type: "distilbert" })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const result = await response.json();
                    displayResult(result);
                } catch (error) {
                    console.error('Error:', error);
                    alert('Error classifying text: ' + error.message);
                } finally {
                    // Reset button state
                    classifyBtn.disabled = false;
                    classifyBtn.textContent = '🔍 Classify Text';
                    loading.style.display = 'none';
                }
            }
            
            function displayResult(result) {
                const resultDiv = document.getElementById('result');
                const headerDiv = document.getElementById('predictionHeader');
                const confidenceDiv = document.getElementById('confidenceText');
                const barsDiv = document.getElementById('probabilityBars');
                
                // Display prediction and confidence
                headerDiv.textContent = `Predicted Category: ${result.prediction}`;
                confidenceDiv.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
                
                // Create probability bars
                let barsHTML = '';
                const sortedProbs = Object.entries(result.probabilities)
                    .sort(([,a], [,b]) => b - a);
                
                sortedProbs.forEach(([className, prob]) => {
                    const percentage = (prob * 100).toFixed(1);
                    const isWinner = className === result.prediction;
                    const barColor = isWinner ? 'linear-gradient(90deg, #28a745, #20c997)' : 'linear-gradient(90deg, #6c757d, #495057)';
                    
                    barsHTML += `
                        <div class="prob-item">
                            <div class="prob-label">
                                <span style="font-weight: ${isWinner ? 'bold' : 'normal'}; color: ${isWinner ? '#28a745' : '#333'}">${className}</span>
                                <span>${percentage}%</span>
                            </div>
                            <div class="prob-bar">
                                <div class="prob-fill" style="width: ${percentage}%; background: ${barColor};">
                                    ${parseFloat(percentage) > 15 ? percentage + '%' : ''}
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                barsDiv.innerHTML = barsHTML;
                resultDiv.style.display = 'block';
                resultDiv.scrollIntoView({ behavior: 'smooth' });
            }
            
            function clearText() {
                document.getElementById('textInput').value = '';
                document.getElementById('result').style.display = 'none';
            }
            
            function setExample(exampleText) {
                document.getElementById('textInput').value = exampleText;
                document.getElementById('result').style.display = 'none';
            }
            
            // Allow Enter key to submit
            document.getElementById('textInput').addEventListener('keydown', function(event) {
                if (event.ctrlKey && event.key === 'Enter') {
                    classify();
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    """Predict text classification"""
    try:
        result = predict_distilbert(input_data.text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Batch prediction from CSV file"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if 'text' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'text' column")
        
        results = []
        for text in df['text']:
            result = predict_distilbert(str(text))
            results.append(result)
        
        return {"predictions": results, "total_processed": len(results)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": distilbert_model is not None,
        "device": str(device)
    }

@app.get("/models/status")
async def model_status():
    """Check model loading status"""
    return {
        "distilbert_loaded": distilbert_model is not None,
        "device": str(device),
        "classes": processor.class_names
    }

if __name__ == "__main__":
    print("🚀 Starting AG News Classifier API...")
    print("📱 Access the web interface at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("💡 Press Ctrl+C to stop the server")
    
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000,
        log_level="info"
    )