# 🌿 Plant Disease Detection + RAG System

An end-to-end crop disease diagnosis system that combines deep learning image classification with a Retrieval-Augmented Generation (RAG) pipeline. Upload a leaf image → get disease diagnosis + treatment recommendations + Q&A support.

---

## 🎯 What It Does

```
User uploads leaf image
        ↓
CNN classifies disease (38 classes, 14 crop species)
        ↓
RAG pipeline retrieves treatment info from knowledge base
        ↓
Groq LLM generates structured treatment response
        ↓
User can ask follow-up questions about the disease
```

---

## 🏗️ Architecture

```
┌─────────────────┐     HTTP      ┌──────────────────────────────────┐
│  Streamlit UI   │ ──────────►  │         FastAPI Backend           │
│  (app.py)       │ ◄──────────  │         (api/main.py)             │
│                 │              │                                    │
│ • Image upload  │              │  ┌─────────────┐  ┌────────────┐ │
│ • Results view  │              │  │  CNN Model  │  │    RAG     │ │
│ • Q&A input     │              │  │EfficientNetB3  │  Pipeline  │ │
└─────────────────┘              │  │ + Custom Head  │            │ │
                                 │  └─────────────┘  └────────────┘ │
                                 └──────────────────────────────────┘
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Dataset | PlantVillage |
| Total Images | 87,000+ |
| Classes | 38 (26 diseases + 12 healthy) |
| Crop Species | 14 |
| Train Split | ~70,000 images |
| Val Split | ~17,000 images |

**14 Crop Species:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

---

## 🧠 Model

### Phase 1 — Scratch CNN
Built from scratch to understand fundamentals:
- 5 convolutional blocks (32 → 64 → 128 → 256 → 512 filters)
- Dense(1500) + Dropout(0.4) + Dense(38, Softmax)
- Input: 128×128 RGB images
- **Limitation:** Struggled on real-world images due to domain gap (trained on lab images only)

### Phase 2 — Transfer Learning (EfficientNetB3)
Upgraded after discovering the domain gap problem:

```
Input (300×300×3)
        ↓
EfficientNetB3 (frozen — 10.78M params)   ← pretrained ImageNet feature extractor
        ↓
GlobalAveragePooling2D
        ↓
BatchNormalization
        ↓
Dense(256, ReLU)                           ← custom classification head
        ↓
Dropout(0.4)
        ↓
Dense(38, Softmax)                         ← 38 disease classes
```

| Metric | Value |
|---|---|
| Val Accuracy | **97.33%** |
| Trainable Params (Phase 1) | 406,310 (3.6%) |
| Total Params | 11,192,917 |
| Training | Google Colab T4 GPU |
| Best Epoch | 11 |

---

## 🔍 RAG Pipeline

| Component | Tool | Details |
|---|---|---|
| Knowledge Base | Groq (llama-3.3-70b) | 38 disease documents, 11 sections each |
| Chunking | Section-based | 418 total chunks |
| Embedding | sentence-transformers | all-MiniLM-L6-v2, 384 dimensions |
| Vector Store | FAISS | IndexFlatL2, local |
| LLM Generation | Groq API | llama-3.3-70b-versatile |

**Each disease document covers:**
- Symptoms, Causes, Favorable Conditions
- Chemical Treatment, Organic Treatment
- Prevention, Severity, Additional Info

**Cost: ₹0** — all components run locally or on free tiers

---

## 🛠️ Tech Stack

```
Deep Learning   : TensorFlow 2.21, Keras 3.4.1, EfficientNetB3
RAG             : FAISS, sentence-transformers, Groq (llama-3.3-70b)
Backend         : FastAPI, Uvicorn
Frontend        : Streamlit
Environment     : Python 3.10, Conda
Training        : Google Colab (T4 GPU)
```

---

## 📁 Project Structure

```
plant-disease-detection-rag/
├── app.py                          ← Streamlit frontend
├── requirements.txt
├── README.md
├── .env.example                    ← Copy to .env and add your keys
│
├── model/
│   ├── trained_model.h5            ← Scratch CNN weights
│   └── saved/
│       └── best_model.h5           ← EfficientNetB3 weights (download separately)
│
├── rag/
│   ├── build_knowledge_base.py     ← Generate disease docs via Groq
│   ├── ingest.py                   ← Chunk + embed + store in FAISS
│   ├── retriever.py                ← Query pipeline
│   ├── knowledge_base/             ← 38 disease .txt files
│   └── embeddings/                 ← FAISS index (generated locally)
│
├── api/
│   └── main.py                     ← FastAPI backend
│
└── notebooks/
    ├── exploration.ipynb
    └── Train_plant_disease.ipynb
```

---

## 🚀 Setup & Running

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/plant-disease-detection-rag.git
cd plant-disease-detection-rag
```

### 2. Create conda environment
```bash
conda create -n plantdisease python=3.10
conda activate plantdisease
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

### 4. Build RAG knowledge base
```bash
cd rag
python build_knowledge_base.py   # generates 38 disease docs
python ingest.py                 # embeds + stores in FAISS
```

### 5. Download model weights
Download `best_model.h5` and place in `model/saved/`
*(Link in releases or Google Drive)*

### 6. Run the system

**Terminal 1 — FastAPI backend:**
```bash
cd api
uvicorn main:app --reload
# Runs on http://localhost:8000
```

**Terminal 2 — Streamlit frontend:**
```bash
cd ..
streamlit run app.py
# Opens http://localhost:8501
```

---

## 🔌 API Endpoints

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `GET /` | GET | — | API status |
| `POST /predict` | POST | Leaf image file | disease, confidence%, treatment |
| `POST /ask` | POST | disease + question | RAG-generated answer |

**Example response from `/predict`:**
```json
{
  "disease": "Potato___Early_blight",
  "confidence": 91.2,
  "treatment": "**Disease Confirmation:** Potato Early Blight caused by Alternaria solani...\n**Treatment:** Apply chlorothalonil or mancozeb fungicide...\n**Prevention:** Crop rotation, remove infected debris..."
}
```

---

## 💡 Key Design Decisions

**Why EfficientNetB3?**
Disease classification is texture-sensitive. EfficientNetB3's compound scaling (width + depth + resolution) captures fine-grained leaf texture at 300×300 resolution better than MobileNetV2 or ResNet alternatives.

**Why two-phase training?**
The head starts with random weights. Unfreezing the backbone immediately would let garbage gradients corrupt 10M pretrained weights. Phase 1 stabilizes the head first, making fine-tuning safe.

**Why section-based chunking?**
Splitting knowledge base by section (SYMPTOMS, TREATMENT, etc.) ensures each chunk is semantically coherent, improving retrieval precision over fixed character-count chunking.

**Why FAISS over Pinecone?**
For 418 vectors, local FAISS is faster (no network latency), free, and sufficient. Pinecone adds cost and complexity without benefit at this scale.

---

## 📈 Results

| Model | Val Accuracy | Notes |
|---|---|---|
| Scratch CNN | — | Good on lab images, poor on real-world |
| EfficientNetB3 (Phase 1) | **97.33%** | 17,572 val images, 38 classes |

*Val set used only for EarlyStopping/checkpointing — not for weight updates*

---

## 🔮 Future Work

- [ ] Phase 2 fine-tuning (top 30 backbone layers)
- [ ] Deploy to cloud (Render / HuggingFace Spaces)
- [ ] Add PlantDoc real-world images to training
- [ ] Top-3 predictions display in UI
- [ ] Confidence threshold warning for low-confidence predictions
- [ ] Multi-language support for treatment recommendations

---

## 📄 License

MIT License

---

## 🙏 Acknowledgements

- [PlantVillage Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019
- [Groq](https://groq.com) — LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI Research
