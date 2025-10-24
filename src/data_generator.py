""" 
outputs .txt files + ground_truth.json 
"""

from __future__ import annotations
import json
import os
import random
import string
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

# -----------------------
# Configuration defaults
# -----------------------

DEFAULT_OUTPUT_DIR = "data/synthetic_texts"
DEFAULT_NUM_PEOPLE = 12
DEFAULT_NUM_DOCS = 12
DEFAULT_SEED = 42

# Generic, fictional vocab pools (intentionally new vs earlier examples)
FIRST_NAMES = [
    "Maya", "Evan", "Priya", "Jonas", "Clara", "Noah",
    "Lena", "Arjun", "Iris", "Kai", "Naomi", "Victor",
    "Tara", "Felix", "Nora", "Zane", "Aisha", "Rowan"
]
LAST_NAMES = [
    "Park", "Rivers", "Sharma", "Cole", "Nguyen", "Miles",
    "Katz", "Ibrahim", "Lopez", "Quinn", "Jensen", "Kaur",
    "Ortiz", "Liu", "Mehta", "Owen"
]
ROLES = [
    "data analyst", "project manager", "software engineer",
    "product designer", "research scientist", "solutions consultant",
    "operations coordinator", "business analyst"
]
COMPANIES = [
    "ApexTech", "Orion Systems", "NovaWorks", "BlueRiver Labs",
    "Crescent Analytics", "Vertex Dynamics"
]
UNIVERSITIES = [
    "Northbridge University", "Eastvale Institute of Technology",
    "Silverridge College", "Westport Polytechnic"
]
LOCATIONS = [
    "Riverton", "Lakeview", "Brookfield", "Northport",
    "Stonehaven", "Maple Grove", "Kingsford"
]
QUALITATIVE_TRAITS = [
    "systematic", "curious", "empathetic", "resilient",
    "decisive", "collaborative", "meticulous", "inventive",
    "outspoken", "diplomatic", "pragmatic", "reflective"
]

# Two/three sentence templates with coreference and varied phrasing
TEMPLATES = [
    (
        "{name} is a {trait1} and {trait2} {role} at {company}. "
        "{pronoun_cap} studied at {university} and now lives in {location}. "
        "{pronoun_cap} often {activity} with colleagues."
    ),
    (
        "At {company}, {name} works as a {role} known for being {trait1}. "
        "After graduating from {university}, {pronoun} settled in {location}. "
        "Peers say {pronoun} is notably {trait2}."
    ),
    (
        "{name}, a {role} at {company}, is considered {trait1}. "
        "{pronoun_cap} previously attended {university} and resides in {location}. "
        "In team settings, {pronoun} tends to be {trait2}."
    ),
]

TEAM_ACTIVITIES = [
    "leads planning sessions", "mentors juniors", "coordinates sprints",
    "prototypes new features", "analyzes datasets", "drafts design specs"
]

# -----------------------
# Data classes
# so that every person or entity follows a consistent schema, making it easy to manipulate / extend later
# -----------------------

@dataclass
class Person:
    pid: str
    name: str
    role: str
    company_id: str
    university_id: str
    location_id: str
    gender: str 
    big_five: Dict[str, float]
    traits: List[str]

@dataclass
class Entity:
    id: str
    type: str  # PERSON / ORG / LOC
    name: str
    attrs: Dict[str, str]

# -----------------------
# Helpers
# -----------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _rand_name(firsts: List[str], lasts: List[str], used: set) -> str:
    # Ensure unique full names
    while True:
        name = f"{random.choice(firsts)} {random.choice(lasts)}"
        if name not in used:
            used.add(name)
            return name

def _pronouns(gender: str) -> Tuple[str, str]:
    if gender == "f":
        return "she", "She"
    return "he", "He"

def _random_big_five() -> Dict[str, float]:
    # Generate constrained random scores with mild correlation
    # (simple approach: random, rounded to 2 decimals)
    def r(): return round(random.uniform(0.15, 0.95), 2)
    return {
        "openness": r(),
        "conscientiousness": r(),
        "extraversion": r(),
        "agreeableness": r(),
        "neuroticism": r()
    }

def _pick_two_distinct(pool: List[str]) -> Tuple[str, str]:
    a, b = random.sample(pool, 2)
    return a, b

def _id(prefix: str, n: int) -> str:
    return f"{prefix}{n:03d}"

# -----------------------
# Core generation
# -----------------------

def _random_big_five() -> Dict[str, float]:
    """Generate mildly correlated random Big Five scores."""
    def r(): return round(random.uniform(0.2, 0.9), 2)
    return {
        "openness": r(),
        "conscientiousness": r(),
        "extraversion": r(),
        "agreeableness": r(),
        "neuroticism": r()
    }

def _traits_from_bigfive(b5: Dict[str, float]) -> List[str]:
    """Derive 2–3 descriptive adjectives based on dominant Big Five scores."""
    traits = []

    # Define associations between Big Five and trait words
    mapping = {
        "openness": [("curious", 0.7), ("inventive", 0.8), ("reflective", 0.6)],
        "conscientiousness": [("organized", 0.7), ("meticulous", 0.8), ("pragmatic", 0.6)],
        "extraversion": [("outspoken", 0.7), ("energetic", 0.8), ("sociable", 0.6)],
        "agreeableness": [("empathetic", 0.7), ("diplomatic", 0.8), ("cooperative", 0.6)],
        "neuroticism": [("resilient", 0.3), ("anxious", 0.8)]  # low N = resilient
    }

    # Choose adjectives based on thresholds
    for trait, candidates in mapping.items():
        val = b5[trait]
        for word, threshold in candidates:
            if (trait == "neuroticism" and val <= threshold and word == "resilient") or (
                trait != "neuroticism" and val >= threshold
            ):
                traits.append(word)
                break  # only take one per dimension

    # Ensure at least 2–3 adjectives
    if len(traits) < 2:
        traits += random.sample(["curious", "organized", "empathetic", "resilient"], 2)
    return traits[:3]


def _make_roster(n_people: int) -> Tuple[List[Person], Dict[str, Entity]]:
    """
    Create people + supporting entities (orgs/unis/locs).
    Returns:
      - people: list of Person
      - entities: dict of id -> Entity for ORG/LOC (people added later)
    """
    entities: Dict[str, Entity] = {}

    # Precreate ORGs / UNIs / LOCs so people can reference them
    org_ids = []
    for i, company in enumerate(COMPANIES, start=1):
        eid = _id("o", i)
        org_ids.append(eid)
        entities[eid] = Entity(id=eid, type="ORG", name=company, attrs={"category": "company"})

    uni_ids = []
    for i, uni in enumerate(UNIVERSITIES, start=1):
        eid = _id("u", i)
        uni_ids.append(eid)
        entities[eid] = Entity(id=eid, type="ORG", name=uni, attrs={"category": "university"})

    loc_ids = []
    for i, loc in enumerate(LOCATIONS, start=1):
        eid = _id("l", i)
        loc_ids.append(eid)
        entities[eid] = Entity(id=eid, type="LOC", name=loc, attrs={})

    used_names = set()
    people: List[Person] = []

    for i in range(1, n_people + 1):
        name = _rand_name(FIRST_NAMES, LAST_NAMES, used_names)
        gender = random.choice(["f", "m"])
        role = random.choice(ROLES)

        company_id = random.choice(org_ids)
        university_id = random.choice(uni_ids)
        location_id = random.choice(loc_ids)

        big_five = _random_big_five()
        traits = _traits_from_bigfive(big_five)


        pid = _id("p", i)
        person = Person(
            pid=pid,
            name=name,
            role=role,
            company_id=company_id,
            university_id=university_id,
            location_id=location_id,
            gender=gender,
            big_five=big_five,
            traits=traits
        )
        people.append(person)

    return people, entities

def _make_relations(people: List[Person]) -> List[Tuple[str, str, str]]:
    """
    Create base relations (works_for, studied_at, lives_in) per person,
    and sprinkle in optional collaborates_with / reports_to edges.
    """
    triples: List[Tuple[str, str, str]] = []

    # Base relations
    for p in people:
        triples.append((p.pid, "works_for", p.company_id))
        triples.append((p.pid, "studied_at", p.university_id))
        triples.append((p.pid, "lives_in", p.location_id))

    # Optional cross-person relations for richness
    # collaborates_with: symmetric
    # reports_to: directional (avoid self)
    pids = [p.pid for p in people]
    if len(pids) >= 4:
        for _ in range(len(pids) // 2):
            a, b = random.sample(pids, 2)
            triples.append((a, "collaborates_with", b))
            triples.append((b, "collaborates_with", a))

        # a few reporting links
        for _ in range(max(1, len(pids) // 6)):
            a, b = random.sample(pids, 2)
            triples.append((a, "reports_to", b))

    return triples

def _person_doc(person: Person, entities: Dict[str, Entity]) -> Tuple[str, List[Dict[str, str]]]:
    """
    Render one person's bio into a short multi-sentence paragraph and
    collect surface-form mentions for doc_index.
    """
    company = entities[person.company_id].name
    university = entities[person.university_id].name
    location = entities[person.location_id].name
    pronoun, pronoun_cap = _pronouns(person.gender)
    trait1, trait2 = person.traits[:2]
    role = person.role
    activity = random.choice(TEAM_ACTIVITIES)

    template = random.choice(TEMPLATES)
    text = template.format(
        name=person.name,
        trait1=trait1,
        trait2=trait2,
        role=role,
        company=company,
        university=university,
        location=location,
        activity=activity,
        pronoun=pronoun,
        pronoun_cap=pronoun_cap
    )

    mentions = [
        {"surface": person.name, "ent_id": person.pid},
        {"surface": company, "ent_id": person.company_id},
        {"surface": university, "ent_id": person.university_id},
        {"surface": location, "ent_id": person.location_id},
    ]
    return text, mentions

def _assemble_ground_truth(
    people: List[Person],
    base_entities: Dict[str, Entity],
    triples: List[Tuple[str, str, str]],
    doc_mentions: Dict[str, List[Dict[str, str]]]
) -> Dict:
    """
    Build the ground truth JSON structure.
    """
    entities: List[Dict] = []

    # Add people as entities
    for p in people:
        entities.append({
            "id": p.pid,
            "type": "PERSON",
            "name": p.name,
            "attrs": {
                "role": p.role,
                "works_for": p.company_id,
                "studied_at": p.university_id,
                "lives_in": p.location_id
            }
        })

    # Add orgs/locs
    for e in base_entities.values():
        entities.append({
            "id": e.id,
            "type": e.type,
            "name": e.name,
            "attrs": e.attrs
        })

    personality = {
    p.pid: {
        "name": p.name,
        "big_five": p.big_five,
        "traits": p.traits
    } for p in people
}


    doc_index = {
        fname: {"mentions": mentions} for fname, mentions in doc_mentions.items()
    }

    gt = {
        "entities": entities,
        "relations": [[s, r, o] for (s, r, o) in triples],
        "personality": personality,
        "doc_index": doc_index
    }
    return gt

# -----------------------
# Public API
# -----------------------

def generate_synthetic_corpus(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    n_people: int = DEFAULT_NUM_PEOPLE,
    n_docs: int = DEFAULT_NUM_DOCS,
    seed: int = DEFAULT_SEED
) -> Dict:
    """
    Generate synthetic bios (.txt) and ground truth JSON.

    Args:
        output_dir: folder to write doc_XXX.txt and ground_truth.json
        n_people: number of unique people to create
        n_docs: number of documents to write (typically == n_people; can be <= n_people)
        seed: RNG seed for determinism

    Returns:
        ground_truth dict (also written to disk)
    """
    random.seed(seed)
    _ensure_dir(output_dir)

    # 1) Roster + entities
    people, base_entities = _make_roster(n_people)

    # 2) Relations
    triples = _make_relations(people)

    # 3) Write docs and collect mentions
    doc_mentions: Dict[str, List[Dict[str, str]]] = {}
    for i, person in enumerate(people[:n_docs], start=1):
        text, mentions = _person_doc(person, base_entities)
        fname = f"doc_{i:03d}.txt"
        with open(os.path.join(output_dir, fname), "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")
        doc_mentions[fname] = mentions

    # 4) Assemble ground truth
    gt = _assemble_ground_truth(people, base_entities, triples, doc_mentions)

    # 5) Write ground_truth.json
    with open(os.path.join(output_dir, "ground_truth.json"), "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)

    return gt


if __name__ == "__main__":
    # Quick manual run
    generate_synthetic_corpus()
    print(f"Synthetic data written to: {DEFAULT_OUTPUT_DIR}")
