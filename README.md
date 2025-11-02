# 🧠 AG-News Topic Classifier using BiLSTM and DistilBERT

Complete pipeline for text classification on AG News dataset using Deep Learning.

## 🎯 FINAL RESULTS SUMMARY
========================================================
📊 **Dataset:** AG News Classification  
🔢 **Classes:** 4 (World, Sports, Business, Science/Technology)  
📈 **Training samples:** 120,000  
🔍 **Test samples:** 7,600  
💻 **Device:** Tesla T4 GPU (15.8 GB)  

### 🏆 **Model Performance**

| Model | Test Accuracy | Architecture | Parameters | Training Epochs |
|-------|---------------|--------------|------------|-----------------|
| **🤖 DistilBERT** | **93.51%** | DistilBERT-base-uncased fine-tuned | ~66M | 5 |
| **🧠 BiLSTM** | **91.33%** | BiLSTM + Attention | 1,664,453 | 5 |

**🏆 Best Model: DistilBERT**  
**🎯 Performance Difference: 2.18%**

---

## 📊 Detailed Classification Reports

### DistilBERT Classification Report:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| World | 0.96 | 0.93 | 0.94 | 1900 |
| Sports | 0.98 | 0.99 | 0.98 | 1900 |
| Business | 0.91 | 0.89 | 0.90 | 1900 |
| Science/Technology | 0.89 | 0.93 | 0.91 | 1900 |
| **Overall** | **0.94** | **0.94** | **0.94** | **7600** |

### BiLSTM Classification Report:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| World | 0.93 | 0.92 | 0.92 | 1900 |
| Sports | 0.96 | 0.98 | 0.97 | 1900 |
| Business | 0.86 | 0.89 | 0.88 | 1900 |
| Science/Technology | 0.90 | 0.86 | 0.88 | 1900 |
| **Overall** | **0.91** | **0.91** | **0.91** | **7600** |

---

## 🔮 Sample Predictions

| Text | DistilBERT | BiLSTM | Agreement |
|------|------------|--------|-----------|
| "Apple Inc. reports record quarterly earnings with strong iPhone sales" | Science/Technology (0.737) | Business (0.567) | ❌ |
| "Scientists discover new exoplanet using advanced telescope technology" | World (0.508) | Science/Technology (1.000) | ❌ |
| "Lakers defeat Warriors in overtime thriller at Staples Center" | Sports (1.000) | Sports (0.997) | ✅ |
| "Breaking: Political tensions rise as world leaders meet for summit" | World (0.986) | World (0.998) | ✅ |

---

## 🧩 Objectives
1. ✅ Apply Deep Learning methods to an NLP problem
2. ✅ Design and implement an intelligent text classifier for news articles
3. ✅ Compare sequence-based (BiLSTM) and transformer-based (DistilBERT) architectures
4. ✅ Build an interactive app for real-time predictions

---

## 🚀 Project Scope

### 🔹 Input & Output Interfaces
**Input:**
- Single text input (headline or short paragraph)
- CSV file upload (column: `text`)

**Output:**
- Predicted category: *World, Sports, Business, or Science/Technology*
- Confidence scores (softmax probabilities)
- Attention visualization

### 🔹 Dataset Details (Actual)
- **Dataset:** [AG News Corpus](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- **Train samples:** 120,000 (used full dataset)
- **Test samples:** 7,600
- **Average text length:** 227 characters
- **Max text length:** 951 characters
- **Classes (4):**
  1. World (25%)
  2. Sports (25%)  
  3. Business (25%)
  4. Science/Technology (25%)

---

## 🧠 Deep Learning Techniques Implemented

### **1️⃣ BiLSTM + Attention**
- **Vocabulary Size:** 10,000 words
- **Embedding Dimension:** 100
- **Hidden Dimension:** 128
- **Layers:** 2 bidirectional LSTM layers
- **Attention Mechanism:** Linear attention for word importance
- **Dropout:** 0.3
- **Optimizer:** Adam (lr=1e-3)

### **2️⃣ DistilBERT Fine-Tuning**
- **Model:** `distilbert-base-uncased` from Hugging Face
- **Architecture:** DistilBERT + Linear classification head
- **Optimizer:** AdamW (lr=2e-5)
- **Batch Size:** 16 (GPU) / 8 (CPU)
- **Max Sequence Length:** 128 tokens
- **Loss:** Cross-Entropy

---

## ⚙️ Implementation Details

| Component | Tools/Frameworks Used |
|-----------|----------------------|
| **Language** | Python 3.10 |
| **Deep Learning** | PyTorch 2.0+ |
| **Transformers** | Hugging Face Transformers |
| **Data Processing** | Pandas, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Google Colab (Tesla T4 GPU) |
| **Notebook** | Jupyter (AG_News_Classification.ipynb) |

---

## 🚀 Quick Start

### 📋 Prerequisites
- Python 3.8+
- PyTorch 2.0+
- GPU recommended (Tesla T4 used in results)

### 🔧 Installation
```bash
# Clone the repository
git clone https://github.com/your-username/agnews-nlp-deeplearning.git
cd agnews-nlp-deeplearning

# Create conda environment
conda env create -f environment.yml
conda activate agnews
```

### 📊 Dataset Setup
1. Download AG News from [Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
2. Place `train.csv` and `test.csv` in `data/` directory

### 🏃‍♂️ Training Models

**Option 1: Jupyter Notebook (Recommended)**
```bash
# Open the comprehensive notebook
jupyter notebook AG_News_Classification.ipynb
# Run all cells for complete pipeline with visualizations
```

**Option 2: Google Colab**
```python
# Upload notebook to Colab
# Install dependencies
!pip install transformers datasets accelerate -q

# Upload data files or mount Google Drive
from google.colab import files
uploaded = files.upload()  # Upload train.csv and test.csv
```

**Option 3: Individual Scripts**
```bash
# Train DistilBERT
python train_distilbert.py --epochs 5 --batch_size 16

# Train BiLSTM
python train_bilstm.py --epochs 5 --batch_size 16
```

### 🌐 Launch Web Interface
```bash
python run_api.py
```
Access at: `http://localhost:8000`

---

## 📁 Project Structure
```
agnews-nlp-deeplearning/
├── src/
│   ├── data/              # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── models/            # Model architectures
│   │   ├── __init__.py
│   │   ├── bilstm_attention.py
│   │   └── distilbert_classifier.py
│   ├── training/          # Training utilities
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── api/              # FastAPI web interface
│   │   ├── __init__.py
│   │   └── app.py
│   └── utils/            # Explainability tools
│       ├── __init__.py
│       └── explainer.py
├── data/                 # Dataset files
│   ├── train.csv
│   └── test.csv
├── models/               # Saved model checkpoints
│   ├── distilbert_model.pth
│   └── bilstm_model.pth
├── assets/screenshots/   # Generated visualizations
├── AG_News_Classification.ipynb  # Complete pipeline notebook
├── print.pdf            # Training results and analysis
├── environment.yml       # Conda environment
├── train_bilstm.py      # BiLSTM training script
├── train_distilbert.py  # DistilBERT training script
├── visualize_results.py # Results analysis
├── run_api.py           # Web interface launcher
└── README.md            # This file
```

---

## 🎯 API Endpoints

### **Main Interface**
- `GET /` - Interactive web interface

### **Prediction**
- `POST /predict` - Single text classification
- `POST /predict/batch` - Batch CSV processing

### **Example Usage**
```python
import requests

response = requests.post("http://localhost:8000/predict", 
    json={"text": "Apple reports record earnings", "model_type": "distilbert"})
print(response.json())
```

---

## 📈 Training Results

### DistilBERT Training (5 Epochs):
```
Epoch 1 - Loss: 0.2378, Val Accuracy: 0.9374
Epoch 2 - Loss: 0.1449, Val Accuracy: 0.9424  
Epoch 3 - Loss: 0.0968, Val Accuracy: 0.9434
Epoch 4 - Loss: 0.0637, Val Accuracy: 0.9414
Epoch 5 - Loss: 0.0432, Val Accuracy: 0.9410
```

### BiLSTM Training (5 Epochs):
```
Epoch 1 - Loss: 0.4155, Val Accuracy: 0.9079
Epoch 2 - Loss: 0.2416, Val Accuracy: 0.9193
Epoch 3 - Loss: 0.1839, Val Accuracy: 0.9190
Epoch 4 - Loss: 0.1368, Val Accuracy: 0.9175
Epoch 5 - Loss: 0.1001, Val Accuracy: 0.9124
```

---

## 🔍 Key Insights

1. **DistilBERT Superior Performance**: 93.51% vs 91.33% accuracy
2. **Sports Classification**: Both models excel (>95% accuracy)
3. **Business Classification**: Most challenging class for both models
4. **Model Agreement**: 50% on sample predictions, showing different strengths
5. **Training Efficiency**: DistilBERT converges faster with better final performance

---

## 📚 References

1. Sanh et al. (2019) — *DistilBERT: A Distilled Version of BERT*
2. Vaswani et al. (2017) — *Attention is All You Need*
3. [AG News Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
4. [Hugging Face Transformers](https://huggingface.co/docs)
5. [PyTorch Documentation](https://pytorch.org/docs)

---

## ✅ Project Status

**All models trained and evaluated successfully!**  
💾 Models saved in `models/` directory  
📊 Ready for deployment or further analysis  
🎯 Achieved excellent accuracy on full AG News dataset  

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.