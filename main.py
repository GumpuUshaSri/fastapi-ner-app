from fastapi import FastAPI
from pydantic import BaseModel
from collections import Counter
import spacy
import re

# Load the English language model (sm = small model, memory-efficient)
try:
    nlp = spacy.load('en_core_web_sm')
    print("SpaCy English language model loaded successfully.")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    nlp = None  # Set nlp to None if the model is not loaded

# Define a Pydantic model for the input text
class TextInput(BaseModel):
    text: str

app = FastAPI()

@app.post("/ner")
def perform_ner(input: TextInput):
    if nlp is None:
        return {"error": "NER model not loaded. Please ensure the model is installed correctly."}

    # Optional text cleaning
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

    # Extract named entities and their labels
    entity_labels = [ent.label_ for ent in doc.ents]
    entity_label_counts = Counter(entity_labels)

    return {
        "entity_summary": dict(entity_label_counts),
        "entities": [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    }
