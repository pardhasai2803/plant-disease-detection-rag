import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rag'))

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
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

# ── Build EfficientNetB3 model ────────────────────────────
def build_model():
    backbone = EfficientNetB3(
        include_top=False,
        weights=None,
        input_shape=(300, 300, 3)
    )
    backbone.trainable = False

    inputs  = tf.keras.Input(shape=(300, 300, 3))
    x       = backbone(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.Dropout(0.4)(x)
    outputs = layers.Dense(38, activation='softmax')(x)

    return models.Model(inputs, outputs)

# ── Load model ────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'saved', 'best_model.h5')
print("Loading EfficientNetB3 model...")
model = build_model()
model.load_weights(MODEL_PATH)
print("Model loaded.")

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

# ── Predict ───────────────────────────────────────────────
def predict_disease(image_bytes):
    img       = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img       = img.resize((300, 300))
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions  = model.predict(img_array, verbose=0)
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
    image_bytes          = await file.read()
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