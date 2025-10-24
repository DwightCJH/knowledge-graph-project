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
    Given the following sentence and list of entities, extract all factual relationships as (subject, predicate, object) triples.

    Rules:
    - Output valid JSON only in this format:
    {{"relations": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}
    - Use the exact entity names as listed.
    - Replace pronouns ("he", "she", "they") with the correct entity name.
    - Use concise, normalized predicates like "works_for", "studied_at", "lives_in", "attended", etc.

    Sentence:
    \"\"\"{sentence}\"\"\"

    Entities:
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


def extract_relations_for_doc(doc_data: dict, model: str = "gpt-4o-mini") -> list:
    """Run relation extraction for one document (multiple sentences)."""
    all_relations = []
    entities = [ent["text"] for ent in doc_data["entities"]]

    for sentence in doc_data["sentences"]:
        rels = extract_relations_from_sentence(sentence, entities, model)
        all_relations.extend(rels)

    return all_relations


def extract_all_relations(input_path="outputs/entities.json", output_path="outputs/relations.json", model="gpt-4o-mini"):
    """Main function to extract relations for all docs."""
    docs = load_json(input_path)
    all_results = {}

    for fname, doc_data in docs.items():
        print(f"🔍 Extracting relations for {fname} ...")
        all_results[fname] = {
            "relations": extract_relations_for_doc(doc_data, model)
        }

    save_json(all_results, output_path)
    print(f"\n✅ Relations saved to {output_path}")
    return all_results


if __name__ == "__main__":
    extract_all_relations()
