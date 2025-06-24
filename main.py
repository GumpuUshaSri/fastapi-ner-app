from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from collections import Counter
import spacy
import pandas as pd
import re

# Load the English language model (sm = small model, memory-efficient)
try:
    nlp = spacy.load('en_core_web_sm')
    print("SpaCy English language model loaded successfully.")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    nlp = None

# Define a Pydantic model for text input
class TextInput(BaseModel):
    text: str

app = FastAPI()

# ✅ Root route
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the FastAPI NER API!",
        "usage": "POST text to /ner or upload a CSV to /upload-csv",
        "docs_url": "/docs"
    }

# ✅ Text-based NER
@app.post("/ner")
def perform_ner(input: TextInput):
    if nlp is None:
        return {"error": "NER model not loaded."}

    def clean_text(text):
        if not isinstance(text, str): return ""
        text = re.sub(r'http\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    cleaned_text = clean_text(input.text)
    doc = nlp(cleaned_text)

    entity_labels = [ent.label_ for ent in doc.ents]
    entity_label_counts = Counter(entity_labels)

    return {
        "entity_summary": dict(entity_label_counts),
        "entities": [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    }

# ✅ CSV upload-based NER
@app.post("/upload-csv")
async def extract_entities_from_csv(file: UploadFile = File(...)):
    if nlp is None:
        return {"error": "NER model not loaded."}
    
    if not file.filename.endswith(".csv"):
        return {"error": "Only CSV files are supported."}

    contents = await file.read()
    try:
        df = pd.read_csv(pd.io.common.BytesIO(contents))
    except Exception as e:
        return {"error": f"Could not read CSV: {str(e)}"}

    if "text" not in df.columns:
        return {"error": "CSV must contain a 'text' column."}

    results = []
    for idx, row in df.iterrows():
        text = row["text"]
        doc = nlp(str(text))
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        results.append({
            "row": idx,
            "original_text": text,
            "entities": entities
        })

    return {"result": results}
