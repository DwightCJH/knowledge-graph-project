"""
evaluation.py
Evaluates entity, relation, and personality extraction accuracy
against ground truth using precision, recall, F1, and MAE metrics.
"""

import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from src.utils import load_json, save_json

# ---------- Utility functions ----------

def precision_recall_f1(true_items, pred_items):
    """Compute precision, recall, and F1 for set-like lists."""
    true_set, pred_set = set(true_items), set(pred_items)
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}


def mean_absolute_error(true_vals, pred_vals):
    """Compute mean absolute error between numeric lists."""
    if not true_vals or not pred_vals:
        return None
    return float(np.mean(np.abs(np.array(true_vals) - np.array(pred_vals))))


def jaccard_index(set1, set2):
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0.0


# ---------- Evaluation Components ----------

def evaluate_entities(gt, pred):
    """Evaluate entity extraction accuracy."""
    results = []
    for doc_name, doc_gt in gt["doc_index"].items():
        gt_entities = {m["surface"].lower() for m in doc_gt["mentions"]}
        pred_entities = {e["text"].lower() for e in pred.get(doc_name, {}).get("entities", [])}
        metrics = precision_recall_f1(gt_entities, pred_entities)
        results.append(metrics)
    avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
    return {"entity_extraction": avg}


def evaluate_relations(gt, pred):
    """Evaluate relation extraction (triple-level matching with ID→name mapping)."""

    # Build lookup tables for ground-truth entity names
    id2name = {e["id"]: e["name"].lower() for e in gt["entities"]}
    
    gt_triples = set()
    for s, p, o in gt["relations"]:
        s_name = id2name.get(s, s).lower()
        o_name = id2name.get(o, o).lower()
        gt_triples.add((s_name, p.lower(), o_name))

    # Build predicted triples set
    pred_triples = set()
    for doc in pred.values():
        for rel in doc["relations"]:
            subj = rel["subject"].lower()
            pred_ = rel["predicate"].lower().replace(" ", "_")
            obj = rel["object"].lower()
            pred_triples.add((subj, pred_, obj))

    metrics = precision_recall_f1(gt_triples, pred_triples)
    return {"relation_extraction": metrics}



def evaluate_personality(gt, pred):
    """Evaluate numeric Big Five and qualitative trait overlap (using surface name mapping)."""
    mae_scores = []
    jaccard_scores = []

    # Ground truth stores: gt["personality"][pid] -> {"name": ..., "big_five": ..., "traits": ...}
    # Pred stores: pred["doc_001.txt"]["Jonas Park"] -> {...}
    for doc_name, doc_pred in pred.items():
        # Iterate over each predicted person in this document
        for pname, pred_traits in doc_pred.items():
            # Try to find a matching ground truth entry by name
            match = None
            for pid, gt_traits in gt["personality"].items():
                if gt_traits.get("name", "").lower() == pname.lower():
                    match = gt_traits
                    break

            if match:
                # Compare numeric Big Five
                true_vals = list(match["big_five"].values())
                pred_vals = list(pred_traits["big_five"].values())
                mae = mean_absolute_error(true_vals, pred_vals)
                mae_scores.append(mae)

                # Compare qualitative adjectives
                jscore = jaccard_index(
                    set(match.get("traits", [])),
                    set(pred_traits.get("traits", []))
                )
                jaccard_scores.append(jscore)

    return {
        "personality_inference": {
            "MAE": float(np.mean(mae_scores)) if mae_scores else None,
            "trait_jaccard": float(np.mean(jaccard_scores)) if jaccard_scores else None
        }
    }

# ---------- Main Evaluation Orchestrator ----------

def evaluate_all(ground_truth_path="data/synthetic_texts/ground_truth.json",
                 entities_path="outputs/entities.json",
                 relations_path="outputs/relations.json",
                 traits_path="outputs/traits.json",
                 output_path="outputs/evaluation_metrics.json"):
    """Evaluate all components and save metrics."""
    gt = load_json(ground_truth_path)
    entities_pred = load_json(entities_path)
    relations_pred = load_json(relations_path)
    traits_pred = load_json(traits_path)
    
    results = {}
    results.update(evaluate_entities(gt, entities_pred))
    results.update(evaluate_relations(gt, relations_pred))
    results.update(evaluate_personality(gt, traits_pred))

    save_json(results, output_path)
    print("✅ Evaluation complete.")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    evaluate_all()
