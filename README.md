⚙️ Setup & Installation

This repository contains a modular Python pipeline for constructing a Knowledge Graph (KG) from synthetic text data.
It performs entity extraction using spaCy, semantic relation and personality inference using an LLM, and builds the graph using NetworkX.

🧩 1. Environment Setup
Python version

✅ Recommended: Python 3.10 – 3.11

These versions ensure compatibility with spaCy 3.7+, PyTorch, and Hugging Face Transformers.

Create and activate a virtual environment
# Create virtual environment
python3 -m venv .venv

# Activate (Mac/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

🧩 2. Install Dependencies
Base dependencies

Install all required packages using:

pip install -r requirements.txt


If you don’t have a requirements.txt yet, these are the core dependencies to include:

spacy==3.7.3
spacy-transformers>=1.3.3
torch>=2.1.0
pandas
networkx
matplotlib
pyvis
pyyaml
openai

🧩 3. Download the spaCy Model
Option 1 — Standard install (works on Intel & most setups)
python -m spacy download en_core_web_trf

Option 2 — Direct wheel install (recommended for M1/M2 Macs)

If the standard command fails, use this direct link:

pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.7.3/en_core_web_trf-3.7.3-py3-none-any.whl


Verify installation:

import spacy
nlp = spacy.load("en_core_web_trf")
print("spaCy model loaded successfully.")

🧩 4. (Optional) Enable Apple Silicon GPU Acceleration

If you are using an M1 or M2 Mac, you can enable GPU acceleration using Apple’s Metal (MPS) backend for PyTorch.

pip install torch torchvision torchaudio


Then verify:

import torch
print(torch.backends.mps.is_available())
# → True means MPS (GPU) backend is active


This is optional — CPU mode already performs well for small synthetic datasets.

🧩 5. Run the Project

From the root directory:

python main.py


Expected output:

🚀 Starting Knowledge Graph pipeline...
✅ Synthetic data generated.
✅ Preprocessing complete. 10 documents processed.
✨ Pipeline test complete. Outputs written to /data and /outputs.

📁 Generated Outputs
Folder	Description
data/synthetic_texts/	Synthetic .txt documents + ground truth JSON
outputs/entities.json	Extracted entities (NER results)
outputs/relations.json	LLM-inferred triples (subject–predicate–object)
outputs/traits.json	Personality inferences (Big Five traits)
outputs/knowledge_graph.gexf	Final KG for visualization in Gephi/PyVis
outputs/evaluation_metrics.json	Evaluation metrics summary
🧠 Notes for Development

Always run scripts from the project root (knowledge_graph_project/), not from inside /src/.

If you modify imports, ensure src/ contains an __init__.py file.

The code is modular — you can run any stage independently (data generation, preprocessing, etc.).

✅ Quick Troubleshooting
Issue	Cause	Fix
ModuleNotFoundError: No module named 'src.utils'	Running from wrong directory	Run python main.py from project root
OSError: [E050] Can't find model 'en_core_web_trf'	Model not installed	Use direct wheel command above
torch.backends.mps.is_available() == False	macOS < 12.3 or old torch	Update macOS / reinstall torch
spacy-transformers not found	Missing plugin	pip install spacy[transformers]
📦 Example Command Summary
# Install dependencies
pip install -r requirements.txt

# Download or install spaCy model
python -m spacy download en_core_web_trf
# or (recommended for Apple Silicon)
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.7.3/en_core_web_trf-3.7.3-py3-none-any.whl

# Verify model load
python -m spacy validate

# Run pipeline
python main.py