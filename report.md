# Knowledge Graph Construction and Personality Modeling from Text

# 1. Introduction

This project implements a complete pipeline to extract entities, relationships, and inferred personality traits from text and assemble them into a structured Knowledge Graph (KG).
It explores the integration of deterministic NLP (via spaCy) and reasoning-based LLM inference to generate structured knowledge and psychological profiles from synthetic text.

## The project delivers:

- A working Python codebase.

- Synthetic dataset with ground truth for evaluation.

- Modular outputs for each stage (entities, relations, traits, KG).

- A final report explaining design rationale and metrics.

# 2. Problem Definition

The goal is to convert unstructured text into structured knowledge representations that capture both factual and personality-related information.
The system should identify entities (people, organizations, locations), extract semantic relations, and infer Big Five personality traits from textual descriptions.

## Key challenges addressed:

- Extracting relations precisely without overgeneration.

- Aligning text and ground truth for meaningful personality evaluation.

- Designing reproducible evaluation metrics for structured and unstructured attributes.

# 3. System Architecture 

The system pipeline processes the text data through the following sequence of components:

1.  **Text Input (`.txt`)** → **Preprocessing (spaCy)**
2.  **Preprocessing (spaCy)** → **Entity Extraction**
3.  **Entity Extraction** → **Relation Extraction (LLM + Filtering)**
4.  **Relation Extraction** → **Personality Inference (LLM)**
5.  **Personality Inference** → **Knowledge Graph Construction (NetworkX)**
6.  **Knowledge Graph Construction** → **Evaluation (Precision, Recall, F1, MAE, Jaccard)**



Each step outputs intermediate artifacts (entities.json, relations.json, traits.json, and .gexf files) for transparency and testing.

# 4. Synthetic Data Generation
## 4.1 Design and Purpose

Synthetic text was generated using templated bios to ensure consistent linguistic structure while maintaining diversity in names, organizations, and roles.
A ground_truth.json file stores canonical entities, relations, and personality data for evaluation.

## 4.2 Key Change 1 – Personality–Text Alignment

Problem:
- Originally, personality scores and traits were assigned randomly, creating a mismatch between text and ground truth.
- The LLM had no real cues to infer Big Five dimensions.

Change Implemented:
- Rewrote data_generator.py to make Big Five scores drive adjective selection and sentence phrasing.
- Traits such as “curious”, “organized”, or “resilient” are now chosen based on corresponding Big Five values.

Rationale:
- This aligns the text’s semantic meaning with numeric personality data, allowing the LLM to infer traits from meaningful evidence rather than randomness.

Outcome:

- Improved MAE (0.229 → 0.2178 → 0.205).

More consistent personality inference and less random variation.

# 5. Preprocessing and Entity Extraction
## 5.1 Implementation

spaCy model: en_core_web_trf for transformer-based NER.

Extracted entity spans, labels, and sentence boundaries.

## 5.2 Reasoning

The transformer model was chosen for higher NER precision despite higher computational cost.

Controlled synthetic data ensures deterministic performance.

## 5.3 Result
| Metric            | Precision | Recall | F1   |
|--------------------|------------|--------|------|
| Entity Extraction  | 1.00       | 1.00   | 1.00 |


Perfect scores validate the preprocessing stage and its reproducibility.

# 6. Relation Extraction
## 6.1 Initial Implementation

Used an LLM prompt to extract subject–predicate–object triples.

Output contained variable phrasing (e.g., “works at”, “resides in”) and inconsistent predicate naming.

Result: low precision due to noise and synonyms.

# 6.2 Key Change 2 – Stricter Relation Schema and Post-Filter

Problem:
- The initial open-ended extraction produced creative but inconsistent relations, such as “is employed by” or “lives at”.
- Pronoun ambiguity also reduced matching accuracy.

Change Implemented:

- Redesigned LLM prompt to restrict output to five fixed predicates: works_for, studied_at, lives_in, collaborates_with, reports_to

- Added a post-filter function (_filter_relations) to:

- Keep only triples whose subject/object exist in the document entities.

- Enforce type constraints (e.g., PERSON → ORG for works_for).

- Remove pronouns and non-canonical edges.

- Normalize predicate names and deduplicate triples.

Rationale:
- Constraining the schema ensures all extracted triples conform to interpretable, valid relation types.
- Filtering eliminates invalid or redundant edges, improving precision at scale.

Outcome:

| Metric                         | Precision | Recall | F1   |
|--------------------------------|------------|--------|------|
| Before (open prompt)           | 0.61       | 0.58   | 0.60 |
| After (strict prompt + filter) | 0.83       | 0.77   | 0.80 |


Precision rose substantially due to the filter removing extraneous triples while maintaining acceptable recall.

# 7. Personality Inference
## 7.1 Initial Version

The LLM generated numeric Big Five scores and adjectives from short bios but without calibration.
This resulted in random scaling and inconsistent vocabulary.

## 7.2 Key Change 3 – Calibrated Prompt with Controlled Vocabulary

Change Implemented:

- Revised the prompt to:

  - Explicitly define score range (0.2 = low, 0.5 = average, 0.8 = high).

  - Restrict adjective output to a fixed approved list.

  - Require strict JSON output with big_five and traits keys.

  - Temperature set to 0 for deterministic outputs.

Rationale:
- LLMs perform better when numeric and lexical spaces are constrained.
- Limiting adjective choice ensures consistent evaluation and reduces synonym noise.

Outcome:

| Metric | Before | After |
| :--- | :--- | :--- |
| MAE | 0.229 → 0.205 | ↓ Improved accuracy |
| Trait Jaccard | 0.64 → 0.63 | ≈ Stable, minor lexical drift |

Interpretation:
- The prompt improved numeric alignment but trait Jaccard remains limited by synonym variation.
- This is an expected trade-off when using a smaller fixed vocabulary.

# 8. Knowledge Graph Construction
## 8.1 Implementation

Constructed using NetworkX with nodes representing entities and people, and edges representing validated relations.

Personality traits stored as node attributes (Big Five + qualitative traits).

Exported to .gexf and .graphml for visualization.

## 8.2 Reasoning

NetworkX chosen over Neo4j for simplicity, no external dependencies, and easy integration with the evaluation framework.

The schema-driven relation extraction ensures type consistency, simplifying KG construction and reducing post-processing.

# 9. Evaluation Framework
## 9.1 Metrics Used
| Component | Metric | Description |
| :--- | :--- | :--- |
| Entity Extraction | Precision, Recall, F1 | Exact name matching |
| Relation Extraction | Precision, Recall, F1 | Triple-level accuracy |
| Personality Inference | MAE | Numeric difference on 0–1 scale |
| | Jaccard | Lexical overlap of adjectives |

## 9.2 Observations

Entity extraction perfect due to deterministic text.

Relation extraction improved significantly through prompt and filter tuning.

Personality inference accuracy improved numerically but remains lexically varied due to synonym drift.

# 10. Final Metrics Summary
| Component | Precision | Recall | F1 / MAE / Jaccard |
| :--- | :--- | :--- | :--- |
| Entity Extraction | 1.00 | 1.00 | 1.00 |
| Relation Extraction | 0.83 | 0.77 | 0.80 |
| Personality Inference (MAE) | — | — | 0.205 |
| Personality Traits (Jaccard) | — | — | 0.63 |

# 11. Analysis of Improvements

## 4. Analysis of Improvements

| Change | Affected Metric | Impact | Reasoning |
| :--- | :--- | :--- | :--- |
| Personality–text alignment in data generation | MAE ↓ | Improves LLM’s ability to infer Big Five from contextual cues | **The data now provides consistent, explicit cues for personality** |
| Stricter predicate schema | Precision ↑ | Removes invalid or vague relations | **Enforces strict entity-type constraints (e.g., Person works\_for Organization)** |
| Relation filtering | Precision ↑ | Ensures type-correct, valid triples | **Post-processing validation removes relations that do not match the expected schema** |
| Calibrated LLM personality prompt | MAE ↓ | Standardizes numeric scaling and lexical output | **Provides the LLM with a clear structure and scale for its output** |
| Controlled vocabulary for traits | Jaccard stable | Reduces unpredictable adjective generation | **Limits the LLM to a predefined set of qualitative adjectives** |

# 12. Limitations

Short templated bios limit expressive variability, constraining personality inference.

Personality Jaccard score is sensitive to lexical choice; synonym normalization is required for further improvement.

Relation schema restricts diversity; new domains may need schema expansion.

LLM inference remains computationally expensive for large datasets.

# 13. Conclusion

This project demonstrates how a hybrid spaCy–LLM pipeline can extract structured knowledge and inferred psychological data from text.
Each design iteration systematically improved precision, interpretability, and consistency.

The relation extraction accuracy (F1 = 0.80) confirms that schema-guided prompting and filtering effectively balance precision and recall.

The personality inference performance (MAE = 0.205) indicates that grounding textual descriptions in personality dimensions yields learnable signals for LLMs.

The modular design allows clear visibility into each step, supporting explainability and reproducibility.

Overall, the pipeline successfully transforms unstructured text into a semantically rich Knowledge Graph while demonstrating measurable, data-driven improvements through deliberate design refinements.

# 14. References

Explosion AI, spaCy NLP Framework. https://spacy.io

OpenAI API, LLM-Based Inference.

NetworkX, Graph Analysis Library.

scikit-learn, Metric Computation Utilities.
