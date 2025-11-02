
# 🧠 AG-News Topic Classifier using BiLSTM and DistilBERT

## 🎯 Aim
To implement a Natural Language Processing (NLP)–based mini project using Deep Learning techniques for automatic text classification.

---

## 🧩 Objectives
1. Apply Deep Learning methods to an NLP problem.
2. Design and implement an intelligent text classifier for news articles.
3. Compare sequence-based (BiLSTM) and transformer-based (DistilBERT) architectures.
4. Build an interactive app for real-time predictions.

---

## 🚀 Project Scope

### 🔹 Input & Output Interfaces
**Input:**
- Single text input (headline or short paragraph)
- CSV file upload (column: `text`)

**Output:**
- Predicted category: *World, Sports, Business, or Science/Technology*
- Confidence scores (softmax probabilities)
- (Optional) Attention or token importance visualization

### 🔹 Dataset Details
- **Dataset:** [AG News Corpus](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- **Train samples:** ~120,000  
- **Test samples:** ~7,600  
- **Classes (4):**
  1. World  
  2. Sports  
  3. Business  
  4. Science/Technology

### 🔹 Major Modules / Functionalities
| Module | Description |
|--------|--------------|
| **Data Loader** | Loads, splits, and prepares AG News dataset. |
| **Preprocessing** | Cleans, tokenizes, and encodes text inputs. |
| **Modeling (BiLSTM)** | Bi-directional LSTM with attention and dense output layer. |
| **Modeling (DistilBERT)** | Transformer encoder fine-tuned on AG News with linear head. |
| **Training & Evaluation** | Training loop with metrics tracking, early stopping, F1-score computation. |
| **Explainability** | Highlights important tokens using attention or LIME/SHAP. |
| **Serving** | Web app using FastAPI for text classification demo. |

---

## 🧹 Text Pre-Processing
| Step | Description |
|------|--------------|
| Lowercasing | Converts text to lowercase (for BiLSTM only). |
| Tokenization | Tokenizes text using TorchText or Hugging Face tokenizer. |
| Cleaning | Removes punctuation, URLs, and extra spaces. |
| Padding/Truncation | Fixes sequence length (e.g., 128 tokens). |
| Encoding | Converts to IDs and attention masks. |
| Split | 80–10–10 train-validation-test split. |

---

## 🧠 Deep Learning Techniques Implemented

### **1️⃣ BiLSTM + Attention**
- Embedding layer (random or pretrained GloVe)
- Bi-directional LSTM
- Attention mechanism for word importance
- Fully connected layer + Softmax

### **2️⃣ DistilBERT Fine-Tuning**
- Pretrained `distilbert-base-uncased` model from Hugging Face
- [CLS] pooled output → Linear classification layer
- Optimizer: AdamW with weight decay
- Scheduler: Linear warmup
- Loss: Cross-Entropy

---

## 🏗️ System Architecture

```text
+-----------------------+
|    AG News Dataset    |
+-----------+-----------+
            |
            v
     [Text Preprocessing]
            |
    +-------+--------+
    |                |
    v                v
 [BiLSTM Model]   [DistilBERT Model]
    |                |
    v                v
  [Training]      [Fine-Tuning]
    \               /
     \             /
      +-----------+
            |
            v
       [Evaluation]
            |
            v
       [Prediction UI]
````

---

## ⚙️ Implementation Details

| Component     | Tools/Frameworks Used                                  |
| ------------- | ------------------------------------------------------ |
| Language      | Python 3.x                                             |
| Libraries     | PyTorch, TorchText, Transformers, Scikit-learn, Pandas |
| Visualization | Matplotlib / Seaborn                                   |
| Interface     | FastAPI                                    `           |
| IDE           | VS Code                                                |
| Hardware      | CPU                                                    |

---

## 📊 Evaluation Metrics

| Metric           | Description                              |
| ---------------- | ---------------------------------------- |
| Accuracy         | Overall correctness                      |
| Precision        | Fraction of correct positive predictions |
| Recall           | Fraction of true positives captured      |
| F1-Score         | Harmonic mean of Precision and Recall    |
| Confusion Matrix | Per-class prediction analysis            |

---

## 📈 Expected Results

| Model                       | Accuracy | F1-Score | Notes            |
| --------------------------- | -------- | -------- | ---------------- |
| **BiLSTM + Attention**      |          |          |                  |
| **DistilBERT (Fine-tuned)** |          |          |                  |


## 📚 References

1. Jurafsky, D. & Martin, J. H. — *Speech and Language Processing* (3rd Edition)
2. Vaswani et al. (2017) — *Attention is All You Need*
3. Sanh et al. (2019) — *DistilBERT: A Distilled Version of BERT*
4. PyTorch Documentation — [https://pytorch.org/docs](https://pytorch.org/docs)
5. Hugging Face Transformers — [https://huggingface.co/docs](https://huggingface.co/docs)
6. Kaggle Dataset — [AG News Classification](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
7. Scikit-learn Metrics — [https://scikit-learn.org/stable/modules/classes.html](https://scikit-learn.org/stable/modules/classes.html)
8. Stanford CS224N Lectures — *YouTube NLP Series*

---

## 🧰 AI Tools & Applications Used

| Tool                            | Purpose                                    |
| ------------------------------- | ------------------------------------------ |
| **Hugging Face Transformers**   | Pretrained DistilBERT model                |
| **PyTorch**                     | Model training and architecture definition |
| **TorchText**                   | Data loading and tokenization for BiLSTM   |
| **Weights & Biases (optional)** | Experiment tracking                        |
| **LIME / SHAP           **      | Model explainability                       |
| **            FastAPI**         | User interface for inference               |
| **ChatGPT**                     | Drafting documentation and design notes    |

---

## 📷 Results & Screenshots

> Add your screenshots in `/assets/screenshots/` and link them below:

| Stage            | Screenshot                                          | Description          |
| ---------------- | --------------------------------------------------- | -------------------- |
| Data Sample      | ![dataset](assets/screenshots/data_sample.png)      | Raw dataset sample   |
| Preprocessing    | ![preprocess](assets/screenshots/preprocess.png)    | Tokenization example |
| Training Curves  | ![training](assets/screenshots/training_curves.png) | Loss/Accuracy graph  |
| Confusion Matrix | ![cm](assets/screenshots/confusion_matrix.png)      | Model evaluation     |
| Streamlit App    | ![ui](assets/screenshots/ui.png)                    | Prediction interface |
| Attention / LIME | ![lime](assets/screenshots/explainability.png)      | Token importance     |

---

## 👥 Contributors

Developed by:

* **[Your Name 1]**
* **[Your Name 2]**
* **[Your Name 3]**

Batch: *[Enter Batch Name / Year]*
Department: *Computer Engineering / AI & DS / IT*
Institution: *[Your College Name]*

---

## 🚀 Quick Start

### 📋 Prerequisites
- Python 3.8+
- PyTorch 2.0+
- GPU recommended (optional)

### 🔧 Installation
```bash
# Clone the repository
git clone https://github.com/patilsrp/agnews-nlp-deeplearning.git
cd agnews-nlp-deeplearning

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate agnews

# Create necessary directories
mkdir models assets/screenshots
```

### 📊 Dataset Setup
1. Download the AG News dataset from [Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
2. Place `train.csv` and `test.csv` in the `data/` directory

### 🏃‍♂️ Training Models

**Train DistilBERT (Recommended):**
```bash
python train_distilbert.py --epochs 5 --batch_size 16 --lr 2e-5
```

**Train BiLSTM + Attention:**
```bash
python train_bilstm.py --epochs 10 --batch_size 32 --lr 1e-3
```

### 🌐 Launch Web Interface
```bash
python run_api.py
```
Access the interactive classifier at: `http://localhost:8000`

### 📈 Analyze Results
```bash
python visualize_results.py
```

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
├── assets/screenshots/   # Generated visualizations
├── environment.yml       # Conda environment specification
├── train_bilstm.py      # BiLSTM training script
├── train_distilbert.py  # DistilBERT training script
├── visualize_results.py # Results analysis
├── run_api.py           # Web interface launcher
└── README.md            # This file
```

---

## 📈 Model Performance

| Model                    | Accuracy | Precision | Recall | F1-Score | Training Time |
|--------------------------|----------|-----------|--------|----------|---------------|
| **BiLSTM + Attention**   | ~88-92%  | ~0.89     | ~0.88  | ~0.88    | ~15-20 min    |
| **DistilBERT**           | ~93-96%  | ~0.94     | ~0.93  | ~0.94    | ~10-15 min    |

*Results may vary based on hardware and hyperparameters*

---

## 🎯 API Endpoints

### **Main Interface**
- `GET /` - Interactive web interface

### **Prediction**
- `POST /predict` - Single text classification
- `POST /predict/batch` - Batch CSV file processing

### **Utilities**
- `GET /models/status` - Check loaded models
- `GET /health` - Health check

### **Example API Usage**
```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict", 
    json={"text": "Apple reports record earnings", "model_type": "distilbert"})
print(response.json())
```

---

## 🔍 Model Explainability Features

### **Attention Visualization**
- View which words the model focuses on
- Available for both BiLSTM and DistilBERT models

### **LIME Explanations**
- Local interpretable model-agnostic explanations
- Highlights positive/negative word contributions

### **Word Clouds**
- Visual representation of important terms
- Generated from model attention/LIME weights

---

## 🛠️ Advanced Usage

### **Custom Training Parameters**
```bash
# DistilBERT with custom settings
python train_distilbert.py \
    --epochs 10 \
    --batch_size 32 \
    --lr 3e-5 \
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --freeze_encoder

# BiLSTM with GloVe embeddings
python train_bilstm.py \
    --epochs 15 \
    --hidden_dim 512 \
    --num_layers 3 \
    --dropout 0.4
```

### **Model Comparison**
```bash
# Generate comprehensive analysis
python visualize_results.py \
    --distilbert_model models/distilbert_model.pth \
    --bilstm_model models/bilstm_model.pth \
    --output_dir results/
```

---

## 📎 Repository

🔗 **GitHub:** [https://github.com/your-username/agnews-nlp-deeplearning](https://github.com/your-username/agnews-nlp-deeplearning)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

