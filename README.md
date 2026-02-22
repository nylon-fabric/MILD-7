# MILD-7
Multi-layer Insight & Linguistic Detection – 7-Level Signal System

## Overview
This project is an analysis system that detects psychological signals from conversational text and uses a 7-point scale.

## Motivation
Developed for the MedGemma Impact Challenge on Kaggle.
This tool is designed to assist mental health professionals
by structuring psychological signals from counseling conversations.


## Features
- Multi-layered structure (prohibitions, emotions, drivers)
- Cosine similarity-based scoring
- 7-level signal output
- Streamlit UI

## Models
- Sentence Transformers – paraphrase-multilingual-MiniLM-L12-v2
- Google MedGemma 1.5 4B IT
- Google Gemma 3 4B IT

## Environment
- Python 3.11
- torch
- transformers
- sentence-transformers
- streamlit

## Installation
```bash
git clone ...
cd ...
pip install -r requirements.txt
```

## Usage
Run the Streamlit application:
```bash
streamlit run app.py
```

## Model Download

Before downloading Gemma / MedGemma models, please ensure you have:

1. Accepted the model license on Hugging Face.   
2. Logged in (if required):   

   huggingface-cli login   

---

### Download Models
```bash
# MiniLM
huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --local-dir ./models/minilm

# MedGemma
huggingface-cli download google/medgemma-1.5-4b-it \
  --local-dir ./models/medgemma

# Gemma
huggingface-cli download google/gemma-3-4b-it \
  --local-dir ./models/gemma
```
### Configure Model Paths
Enter the local model paths in:   
src/settings.py
```python
minilm_url = "./models/minilm"
medgemma_url = "./models/medgemma"
gemma_url = "./models/gemma"
```

### Notes
Models are loaded using local_files_only=True.   
Model is reloaded per inference to avoid VRAM fragmentation under 8GB constraint.   
Recommended GPU memory: 8GB+   

## 🎥 Demo Video
[https://youtu.be/SkzBbA5vVqg](https://youtu.be/SkzBbA5vVqg)
