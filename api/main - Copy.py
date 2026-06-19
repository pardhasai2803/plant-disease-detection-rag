import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rag'))

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from retriever import get_disease_info

# ── App ───────────────────────────────────────────────────
app = FastAPI(title="Crop Disease Diagnosis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load CNN model ────────────────────────────────────────
print("Loading CNN model...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'trained_model.h5')
# Update loading to rebuild + load weights
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Sequential

def build_model():
    model = Sequential()
    model.add(tf.keras.Input(shape=(128, 128, 3)))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(256, 3, padding='same', activation='relu'))
    model.add(Conv2D(256, 3, activation='relu'))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(512, 3, padding='same', activation='relu'))
    model.add(Conv2D(512, 3, activation='relu'))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(38, activation='softmax'))
    return model

print("Loading CNN model...")
model = build_model()
model.load_weights(MODEL_PATH)
print("CNN model loaded.")

# ── Class names ───────────────────────────────────────────
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ── Predict disease from image ────────────────────────────
def predict_disease(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) 
    img_array = np.expand_dims(img_array, axis=0)

    predictions  = model.predict(img_array)
    class_idx    = np.argmax(predictions[0])
    confidence   = float(np.max(predictions[0]))
    disease_name = CLASS_NAMES[class_idx]

    return disease_name, confidence

# ── Routes ────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Crop Disease Diagnosis API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes  = await file.read()
    disease_name, confidence = predict_disease(image_bytes)

    rag_result = get_disease_info(
        disease_name=disease_name,
        user_query=f"What are the symptoms, treatment and prevention for {disease_name}?"
    )

    return {
        "disease"    : disease_name,
        "confidence" : round(confidence * 100, 2),
        "treatment"  : rag_result["response"]
    }

@app.post("/ask")
async def ask(disease: str, question: str):
    rag_result = get_disease_info(
        disease_name=disease,
        user_query=question
    )
    return {
        "disease"  : disease,
        "question" : question,
        "answer"   : rag_result["response"]
    }