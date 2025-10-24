"""
handles text preprocessing and entity extraction using spaCy.

outputs:
- outputs/entities.json
"""

import os
import json
import spacy
from typing import List, Dict
from src.utils import save_json

# Load spaCy model globally (better performance)
NLP_MODEL = "en_core_web_trf"
nlp = spacy.load(NLP_MODEL)

def read_synthetic_texts(folder: str = "data/synthetic_texts") -> Dict[str, str]:
    """Read all .txt files in the synthetic_texts directory."""
    docs = {}
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".txt"):
            path = os.path.join(folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                docs[fname] = f.read().strip()
    return docs


def run_spacy_pipeline(docs: Dict[str, str]) -> Dict[str, Dict]:
    """
    Apply spaCy pipeline (sentence segmentation, NER, dependency parsing)
    and extract structured info for each doc.
    """
    all_results = {}

    for fname, text in docs.items():
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

        sentences = [sent.text.strip() for sent in doc.sents]

        all_results[fname] = {
            "entities": entities,
            "sentences": sentences
        }

    return all_results


def preprocess_texts(input_dir: str = "data/synthetic_texts",
                     output_path: str = "outputs/entities.json") -> Dict[str, Dict]:
    """
    High-level function: read texts, run spaCy, and save entity outputs.
    """
    docs = read_synthetic_texts(input_dir)
    processed = run_spacy_pipeline(docs)
    save_json(processed, output_path)
    return processed


if __name__ == "__main__":
    results = preprocess_texts()
    print(f"Preprocessed {len(results)} documents → outputs/entities.json")
