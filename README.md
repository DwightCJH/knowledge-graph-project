# Dataset
## Synthetic Data Generation

All text is synthetically generated using `data_generator.py`, ensuring reproducibility and complete ground truth availability.
Each text file (e.g., `doc_001.txt`) describes a person, their workplace, education, and location, along with personality cues consistent with underlying Big Five scores.

### Example:

> Jonas Park is a curious and resilient product designer at Orion Systems.
> He studied at Eastvale Institute of Technology and now lives in Maple Grove.
> He often coordinates sprints with colleagues.

### Ground Truth Example (`ground_truth.json`):

```json
{
  "entities": [...],
  "relations": [["p001", "works_for", "o002"], ["p001", "studied_at", "u003"]],
  "personality": {
    "p001": {
      "big_five": {...},
      "traits": ["curious", "resilient"]
    }
  }
}
```

# Setup Instructions

## 1. Environment Setup
Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Configure API and Models
Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your_api_key_here"
```
Install the spaCy transformer model (required for NER and sentence segmentation):
```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.7.3/en_core_web_trf-3.7.3-py3-none-any.whl
```

# Execution
You can run the entire pipeline from start to finish with a single command:
```bash
python main.py
```
This will automatically:

- Generate synthetic text data and ground truth (data/synthetic_texts/).

- Run preprocessing and entity extraction using spaCy.

- Extract relations using the LLM and apply schema-based filtering.

- Infer personality traits (Big Five + qualitative adjectives).

- Construct and export the Knowledge Graph (.gexf, .graphml).

- Evaluate results and save metrics in outputs/evaluation_metrics.json.

# Results
## Evaluation Summary
| Component | Precision | Recall | F1 / MAE / Jaccard |
| :--- | :--- | :--- | :--- |
| **Entity Extraction** | 1.00 | 1.00 | 1.00 |
| **Relation Extraction** | 0.83 | 0.77 | 0.80 |
| **Personality Inference (MAE)** | — | — | 0.205 |
| **Personality Traits (Jaccard)** | — | — | 0.63 |

## Key Findings

- Entity extraction achieved perfect performance due to structured text.

- Relation extraction improved significantly after schema enforcement and post-filtering.

- Personality inference achieved moderate accuracy (MAE = 0.205) and consistent qualitative alignment.
