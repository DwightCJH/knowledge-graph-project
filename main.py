from src.data_generator import generate_synthetic_corpus
from src.preprocessing import preprocess_texts
from src.relation_extraction import extract_all_relations
from src.personality_inference import infer_all_personalities
from src.kg_builder import build_knowledge_graph, visualize_graph
from src.evaluation import evaluate_all


def main():
    print("🚀 Starting Knowledge Graph pipeline...\n")

    # 1️⃣ Generate synthetic data
    gt = generate_synthetic_corpus(
        output_dir="data/synthetic_texts",
        n_people=10,
        n_docs=10,
        seed=42
    )
    print("✅ Synthetic data generated.\n")

    # 2️⃣ Preprocess texts (spaCy)
    processed = preprocess_texts(
        input_dir="data/synthetic_texts",
        output_path="outputs/entities.json"
    )
    print(f"✅ Preprocessing complete. {len(processed)} documents processed.\n")

    # 3️⃣ Extract relations using LLM
    relations = extract_all_relations(
        input_path="outputs/entities.json",
        output_path="outputs/relations.json",
        model="gpt-4o-mini"
    )
    print("✅ Relation extraction complete.\n")

    # 4️⃣ Infer personality traits using LLM
    personalities = infer_all_personalities(
        input_path="outputs/entities.json",
        output_path="outputs/traits.json",
        model="gpt-4o-mini"
    )
    print("✅ Personality inference complete.\n")

    # 5️⃣ Build Knowledge Graph
    G = build_knowledge_graph(
        entities_path="outputs/entities.json",
        relations_path="outputs/relations.json",
        traits_path="outputs/traits.json",
        output_dir="outputs/graphs"
    )
    print("✅ Knowledge Graph construction complete.\n")

    # 6️⃣ Visualize Knowledge Graph
    visualize_graph(
        graph_path="outputs/graphs/knowledge_graph.gexf",
        output_html="outputs/graphs/knowledge_graph.html"
    )
    print("✅ Knowledge Graph visualization complete.\n")

    # 7️⃣ Evaluate all components
    results = evaluate_all()
    print("✅ Evaluation complete.\n")

    print("✨ Full pipeline finished! Outputs written to /data and /outputs.")


if __name__ == "__main__":
    main()
