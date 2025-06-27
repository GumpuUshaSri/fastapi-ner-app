from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from collections import Counter, defaultdict
import spacy
import pandas as pd
import re
import json
import io
from fastapi.responses import PlainTextResponse

# Load SpaCy English model
try:
    nlp = spacy.load('en_core_web_sm')
    print("SpaCy model loaded successfully.")
except OSError:
    print("Model not found. Run 'python -m spacy download en_core_web_sm'")
    nlp = None

app = FastAPI()

# Pydantic model for text input
class TextInput(BaseModel):
    text: str

# Text cleaner
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Shared NER logic
def process_docs(texts):
    entity_counter = Counter()
    entity_groups = defaultdict(set)

    for text in texts:
        cleaned = clean_text(text)
        doc = nlp(cleaned)
        for ent in doc.ents:
            entity_counter[ent.label_] += 1
            entity_groups[ent.label_].add(ent.text)

    summary = {
        label: {
            "count": count,
            "entities": sorted(list(entity_groups[label]))
        } for label, count in entity_counter.items()
    }

    return {"summary": summary}

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "NER API is live!",
        "usage": {
            "Text": "POST to /ner",
            "CSV Upload": "POST to /upload-csv",
            "TXT Summary": "POST to /download-summary-txt"
        },
        "docs_url": "/docs"
    }

# Text-based NER
@app.post("/ner")
def perform_ner(input: TextInput):
    if nlp is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    cleaned = clean_text(input.text)
    doc = nlp(cleaned)
    labels = [ent.label_ for ent in doc.ents]
    counts = Counter(labels)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return {"entity_summary": dict(counts), "entities": entities}

# CSV upload with NER
@app.post("/upload-csv")
async def extract_entities_from_csv(file: UploadFile = File(...)):
    if nlp is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")

    contents = await file.read()
    try:
        df = pd.read_csv(pd.io.common.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV read error: {str(e)}")

    if "text" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain 'text' column.")

    df = df.dropna(subset=["text"])  # Drop rows with NaN in 'text'

    results = []
    for idx, row in df.iterrows():
        text = str(row["text"])
        doc = nlp(text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        results.append({
            "row": idx,
            "original_text": text,
            "entities": entities
        })

    return {"result": results}

# TXT summary download
@app.post("/download-summary-txt")
async def download_summary_txt(file: UploadFile = File(...)):
    if nlp is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")

    contents = await file.read()
    try:
        df = pd.read_csv(pd.io.common.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV read error: {str(e)}")

    if "text" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain 'text' column.")

    texts = df["text"].dropna().astype(str).tolist()
    result = process_docs(texts)
    summary = result["summary"]

    lines = ["NER Summary Report:\n"]
    for label, data in summary.items():
        lines.append(f"{label}: {data['count']} occurrence(s)")
        lines.append(f"  Entities: {', '.join(data['entities'])}\n")

    content = "\n".join(lines)

    return PlainTextResponse(
        content,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=ner_summary.txt"}
    )
