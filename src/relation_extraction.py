"""
Extracts semantic relationships (triples) between entities using an LLM.
"""

import os
import json
import re
from openai import OpenAI
from src.utils import load_json, save_json

# Initialize client (expects environment variable OPENAI_API_KEY)
client = OpenAI()

def extract_relations_from_sentence(sentence: str, entities: list, model: str = "gpt-4o-mini") -> list:
    """Prompt LLM to extract triples from one sentence."""
    prompt = f"""
        You are an information extraction assistant.

        Extract only factual relationships that fit this schema. Do not infer beyond the sentence.

        ALLOWED PREDICATES (use EXACT spelling):
        - works_for
        - studied_at
        - lives_in
        - collaborates_with
        - reports_to

        STRICT RULES:
        - Use ONLY the above predicates. Do NOT invent others (e.g., do NOT use has_trait).
        - The subject and object MUST be copied EXACTLY from the Entities list below. Do NOT invent or alter names. 
        - If the sentence uses pronouns (he, she, they), REPLACE them with the correct full entity name from the Entities list. If you’re unsure, DROP the triple.
        - Enforce type constraints:
        • works_for: PERSON → ORG
        • studied_at: PERSON → ORG
        • lives_in: PERSON → GPE or LOC
        • collaborates_with: PERSON ↔ PERSON
        • reports_to: PERSON → PERSON
        - If no valid triple exists, return an empty list.

        OUTPUT JSON ONLY in this exact format:
        {{
        "relations": [
            {{"subject": "<entity from list>", "predicate": "<one of the allowed>", "object": "<entity from list>"}}
        ]
        }}

        Sentence:
        \"\"\"{sentence}\"\"\"

        Entities (copy names EXACTLY from here):
        {entities}
        """



    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0, #set to 0 for deterministic output -> better for extraction tasks -> more consistent/reproducible results
    )

    content = response.choices[0].message.content.strip()

    # Extract JSON block (handles cases where LLM adds extra text)
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return []

    try:
        data = json.loads(match.group(0))
        return data.get("relations", [])
    except json.JSONDecodeError:
        return []

def _filter_relations(doc_data, relations):
    """
    Precision-focused filter:
    - subject/object must be in the doc's entity surface list
    - enforce type constraints by spaCy labels (PERSON/ORG/GPE)
    - normalize predicate to allowed set
    - drop pronouns if any slipped through
    """
    allowed_preds = {"works_for", "studied_at", "lives_in", "collaborates_with", "reports_to"}

    # Build lookup maps from entities.json for this doc
    name2type = {}
    entity_names = set()
    for e in doc_data["entities"]:
        nm = e["text"]
        entity_names.add(nm)
        name2type[nm] = e["label"]  # PERSON / ORG / GPE / LOC

    def is_pronoun(x: str) -> bool:
        return x.lower() in {"he", "she", "they", "him", "her", "them"}

    def norm_pred(p: str) -> str:
        return p.lower().strip().replace(" ", "_")

    def type_ok(pred, s_name, o_name):
        st = name2type.get(s_name, "")
        ot = name2type.get(o_name, "")
        if pred == "works_for":       return st == "PERSON" and ot == "ORG"
        if pred == "studied_at":      return st == "PERSON" and ot == "ORG"
        if pred == "lives_in":        return st == "PERSON" and ot in {"GPE", "LOC"}
        if pred == "collaborates_with": return st == "PERSON" and ot == "PERSON"
        if pred == "reports_to":      return st == "PERSON" and ot == "PERSON"
        return False

    cleaned = []
    for r in relations:
        subj = r.get("subject", "")
        obj  = r.get("object", "")
        pred = norm_pred(r.get("predicate", ""))

        # Predicate whitelist
        if pred not in allowed_preds:
            continue

        # Reject pronouns if any slipped through
        if is_pronoun(subj) or is_pronoun(obj):
            continue

        # Subject/Object must come exactly from this doc's Entities list
        if subj not in entity_names or obj not in entity_names:
            continue

        # Type constraints
        if not type_ok(pred, subj, obj):
            continue

        cleaned.append({"subject": subj, "predicate": pred, "object": obj})

    return cleaned


def extract_relations_for_doc(doc_data: dict, model: str = "gpt-4o-mini") -> list:
    """Run relation extraction for one document (multiple sentences)."""
    all_relations = []
    entities = [ent["text"] for ent in doc_data["entities"]]

    for sentence in doc_data["sentences"]:
        rels = extract_relations_from_sentence(sentence, entities, model)
        rels = _filter_relations(doc_data, rels)
        all_relations.extend(rels)
        dedup = {(r["subject"], r["predicate"], r["object"]) for r in all_relations}
        all_relations = [{"subject": s, "predicate": p, "object": o} for (s, p, o) in dedup]
    return all_relations


def extract_all_relations(input_path="outputs/entities.json", output_path="outputs/relations.json", model="gpt-4o-mini"):
    """Main function to extract relations for all docs."""
    docs = load_json(input_path)
    all_results = {}

    for fname, doc_data in docs.items():
        print(f"Extracting relations for {fname} ...")
        all_results[fname] = {
            "relations": extract_relations_for_doc(doc_data, model)
        }

    save_json(all_results, output_path)
    print(f"\nRelations saved to {output_path}")
    return all_results


if __name__ == "__main__":
    extract_all_relations()
