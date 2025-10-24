"""
Infers Big Five personality traits for human subjects mentioned in text documents using an LLM.
"""

import json
import re
from openai import OpenAI
from src.utils import load_json, save_json

client = OpenAI()

def infer_personality_from_text(person_name: str, text: str, model: str = "gpt-4o-mini") -> dict:
    """
    Prompt the LLM to infer the Big Five personality traits for a given person
    based on all sentences mentioning or describing them.
    """
    prompt = f"""
    You are a personality analysis assistant. 
    Given the following biographical text about a person, infer their personality traits 
    according to the Big Five model (Openness, Conscientiousness, Extraversion, 
    Agreeableness, and Neuroticism).

    Output valid JSON only in this format:
    {{
      "big_five": {{
        "openness": <float between 0 and 1>,
        "conscientiousness": <float between 0 and 1>,
        "extraversion": <float between 0 and 1>,
        "agreeableness": <float between 0 and 1>,
        "neuroticism": <float between 0 and 1>
      }},
      "traits": ["<adjective1>", "<adjective2>", ...]
    }}

    Person: {person_name}

    Biography text:
    \"\"\"{text}\"\"\"
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return {}

    try:
        data = json.loads(match.group(0))
        return data
    except json.JSONDecodeError:
        return {}


def infer_personalities_for_doc(doc_data: dict, model: str = "gpt-4o-mini") -> dict:
    """
    Infer personality traits for each PERSON entity in the document.
    Aggregates all sentences as input for that person.
    """
    results = {}
    text = " ".join(doc_data["sentences"])

    # Get all PERSON entities
    persons = [ent["text"] for ent in doc_data["entities"] if ent["label"] == "PERSON"]

    # If multiple PERSONs appear, analyze each independently
    for person in persons:
        print(f"🧠 Inferring traits for {person}...")
        traits = infer_personality_from_text(person, text, model)
        if traits:
            results[person] = traits

    return results


def infer_all_personalities(input_path="outputs/entities.json",
                            output_path="outputs/traits.json",
                            model="gpt-4o-mini"):
    """Main entry point to infer Big Five traits for all documents."""
    docs = load_json(input_path)
    all_results = {}

    for fname, doc_data in docs.items():
        print(f"🔍 Inferring personalities in {fname} ...")
        all_results[fname] = infer_personalities_for_doc(doc_data, model)

    save_json(all_results, output_path)
    print(f"\n✅ Personality traits saved to {output_path}")
    return all_results


if __name__ == "__main__":
    infer_all_personalities()
