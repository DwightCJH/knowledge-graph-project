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

    Given a short biography text, estimate the person's Big Five personality traits 
    and list 2–3 descriptive adjectives that best represent those traits.

    ### Big Five traits:
    - openness
    - conscientiousness
    - extraversion
    - agreeableness
    - neuroticism

    ### Rules:
    - Assign a numeric score between 0.0 and 1.0 for each trait.
    Use this guideline:
    0.2 = low, 0.5 = average, 0.8 = high.
    - Select adjectives only from the approved vocabulary below.
    If none fit perfectly, choose the closest match — do NOT invent new words.

    Approved trait vocabulary:
    [curious, inventive, reflective, organized, meticulous, pragmatic, outspoken, energetic, sociable, empathetic, diplomatic, cooperative, resilient, anxious]

    ### Output format (JSON only):
    {{
    "big_five": {{
        "openness": ...,
        "conscientiousness": ...,
        "extraversion": ...,
        "agreeableness": ...,
        "neuroticism": ...
    }},
    "traits": ["...", "..."]
    }}

    Text:
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
        print(f"Inferring traits for {person}...")
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
        print(f"Inferring personalities in {fname} ...")
        all_results[fname] = infer_personalities_for_doc(doc_data, model)

    save_json(all_results, output_path)
    print(f"\n Personality traits saved to {output_path}")
    return all_results


if __name__ == "__main__":
    infer_all_personalities()
