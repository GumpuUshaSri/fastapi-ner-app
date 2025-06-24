from fastapi import FastAPI
from pydantic import BaseModel
from collections import Counter
import spacy
import re

# Load the English language model
try:
    nlp = spacy.load('en_core_web_lg')
    print("SpaCy English language model loaded successfully.")
except OSError:
    print("SpaCy model 'en_core_web_lg' not found. Please run 'python -m spacy download en_core_web_lg' in your terminal.")
    # You might want to handle this error more gracefully in a production app,
    # perhaps by raising an exception or returning an error response.
    nlp = None # Set nlp to None if the model is not loaded

# Define a Pydantic model for the input text
class TextInput(BaseModel):
    text: str

app = FastAPI()

@app.post("/ner")
def perform_ner(input: TextInput):
    if nlp is None:
        return {"error": "NER model not loaded. Please ensure the model is installed correctly."}

    # Access the text data
    text_to_process = input.text

    # Optional: Apply the same cleaning as done in the notebook if needed for API input
    # from here down is where you would put the cleaning logic if you want to apply it to the API input
    # import re
    # def clean_text(text):
    #     if not isinstance(text, str): return ""
    #     text = re.sub(r'http\S+|https\S+', '', text)
    #     text = re.sub(r'@\w+', '', text)
    #     text = re.sub(r'#\w+', '', text)
    #     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    #     text = text.lower()
    #     text = re.sub(r'\s+', ' ', text).strip()
    #     return text
    # cleaned_text = clean_text(text_to_process)


    # Process the text using the loaded spaCy model
    # Use cleaned_text if you uncommented and used the cleaning logic above
    doc = nlp(text_to_process)


    # Extract named entities and their labels
    entity_labels = [ent.label_ for ent in doc.ents]

    # Count the occurrences of each label
    entity_label_counts = Counter(entity_labels)

    # Return the summary of entity counts
    return {"entity_summary": dict(entity_label_counts)}

# To run this application, save the code as main.py and run 'uvicorn main:app --reload' in your terminal