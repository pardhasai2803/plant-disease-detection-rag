import os
import json
import time
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Setup ─────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
KB_DIR = Path("knowledge_base")
KB_DIR.mkdir(exist_ok=True)

# ── All 38 disease classes ────────────────────────────────
DISEASES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# ── Prompt template ───────────────────────────────────────
def build_prompt(disease):
    return f"""You are an agricultural expert. Generate a detailed knowledge base document for the following crop disease or condition: {disease}

Format the response exactly as follows:

DISEASE: {disease}
CROP: [crop name]
TYPE: [fungal/bacterial/viral/healthy]
SYMPTOMS: [detailed symptoms]
CAUSES: [what causes this disease]
FAVORABLE_CONDITIONS: [conditions that favor disease spread]
CHEMICAL_TREATMENT: [chemical treatments and pesticides]
ORGANIC_TREATMENT: [organic/natural treatments]
PREVENTION: [prevention measures]
SEVERITY: [low/medium/high]
ADDITIONAL_INFO: [any other relevant information]

Be specific, detailed and accurate. For healthy plants, describe characteristics of a healthy plant and general care tips."""

# ── Generate documents ────────────────────────────────────
print(f"Generating knowledge base for {len(DISEASES)} diseases...\n")

for i, disease in enumerate(DISEASES):
    # Skip if already generated
    output_file = KB_DIR / f"{disease.replace('/', '_')}.txt"
    if output_file.exists():
        print(f"[{i+1:2d}/38] Already exists — skipping: {disease}")
        continue

    try:
        print(f"[{i+1:2d}/38] Generating: {disease}")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": build_prompt(disease)}],
            max_tokens=1000,
            temperature=0.3
        )

        content = response.choices[0].message.content

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"         ✅ Saved: {output_file.name}")

        # Rate limit — Groq free tier allows 30 req/min
        time.sleep(2)

    except Exception as e:
        print(f"         ❌ Error: {e}")
        time.sleep(5)

print(f"\nDone! Knowledge base saved to {KB_DIR}")
print(f"Total files: {len(list(KB_DIR.glob('*.txt')))}")