"""
Constructs a Knowledge Graph (KG) from extracted entities, relations, and personality traits.
Exports the graph in .gexf, .graphml, and .html formats.
"""

import os
import networkx as nx
from pyvis.network import Network
from src.utils import load_json, save_json, ensure_dir


def build_knowledge_graph(entities_path="outputs/entities.json",
                          relations_path="outputs/relations.json",
                          traits_path="outputs/traits.json",
                          output_dir="outputs/graphs"):
    """Combine extracted data and build the unified Knowledge Graph."""
    
    # Load inputs
    entities_data = load_json(entities_path)
    relations_data = load_json(relations_path)
    traits_data = load_json(traits_path)

    # Initialize graph
    G = nx.DiGraph()

    # Helper function to add nodes safely
    def add_node(name, ntype="UNKNOWN", **attrs):
        if not G.has_node(name):
            G.add_node(name, type=ntype, **attrs)
        else:
            G.nodes[name].update(attrs)

    # Add all entities
    for fname, doc in entities_data.items():
        for ent in doc["entities"]:
            add_node(ent["text"], ntype=ent["label"])

    # Add all relations
    for fname, doc in relations_data.items():
        for rel in doc["relations"]:
            subj = rel["subject"]
            obj = rel["object"]
            pred = rel["predicate"]

            # Handle pronouns
            if subj.lower() in ["he", "she"]:
                persons = [e["text"] for e in entities_data[fname]["entities"] if e["label"] == "PERSON"]
                if persons:
                    subj = persons[0]

            # Add nodes if missing
            if not G.has_node(subj):
                add_node(subj, ntype="UNKNOWN")
            if not G.has_node(obj):
                add_node(obj, ntype="UNKNOWN")

            G.add_edge(subj, obj, label=pred)

    # Add personality traits as node attributes
    for fname, doc_traits in traits_data.items():
        for person, pdata in doc_traits.items():
            if not G.has_node(person):
                add_node(person, ntype="PERSON")

            # Big Five attributes
            for trait, score in pdata["big_five"].items():
                G.nodes[person][trait] = score

            # Trait adjectives as separate nodes
            for tword in pdata.get("traits", []):
                if not G.has_node(tword):
                    add_node(tword, ntype="TRAIT")
                G.add_edge(person, tword, label="has_trait")

    # Export graph
    ensure_dir(output_dir)
    gexf_path = os.path.join(output_dir, "knowledge_graph.gexf")
    graphml_path = os.path.join(output_dir, "knowledge_graph.graphml")

    nx.write_gexf(G, gexf_path)
    nx.write_graphml(G, graphml_path)

    print(f"Knowledge Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"Exported to: {gexf_path}, {graphml_path}")

    return G

def visualize_graph(
    graph_path="outputs/graphs/knowledge_graph.gexf",
    output_html="outputs/graphs/knowledge_graph.html",
    height="1960px",
    trait_to_highlight="extraversion"
):
    """Interactive visualization: click a node to show only its connected neighbors."""
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    G = nx.read_gexf(graph_path)
    net = Network(height=height, width="100%", directed=True, notebook=False)
    net.barnes_hut()

    # Color palette by node type
    type_colors = {
        "PERSON": "#FFB74D",  # orange
        "ORG": "#64B5F6",     # blue
        "LOC": "#81C784",     # green
        "TRAIT": "#BA68C8",   # purple
        "UNKNOWN": "#E0E0E0"  # gray
    }

    # --- Add nodes ---
    for node, data in G.nodes(data=True):
        ntype = data.get("type", "UNKNOWN")
        base_color = type_colors.get(ntype, "#E0E0E0")

        # Highlight intensity by selected Big Five trait
        brightness_factor = 1.0
        if ntype == "PERSON" and trait_to_highlight in data:
            val = float(data[trait_to_highlight])
            brightness_factor = 0.5 + (val * 0.7)

        def adjust_color(hex_color, factor):
            rgb = [int(hex_color[i:i+2], 16) for i in (1, 3, 5)]
            rgb = [min(255, int(c * factor)) for c in rgb]
            return f"#{''.join(f'{c:02x}' for c in rgb)}"

        color = adjust_color(base_color, brightness_factor)

        # Tooltip content
        title_lines = [f"\n{node}\n", f"Type: {ntype}"]
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            if trait in data:
                title_lines.append(f"{trait.capitalize()}: {data[trait]}")
        title = "\n".join(title_lines)

        # Node size scaled by trait
        size = 120 if ntype == "PERSON" else 90
        if ntype == "PERSON" and trait_to_highlight in data:
            size = 90 + float(data[trait_to_highlight]) * 80

        net.add_node(
            node,
            label=node,
            title=title,
            color=color,
            shape="dot",
            size=size,
            font={"size": 40}
        )

    # --- Add edges ---
    for u, v, data in G.edges(data=True):
        net.add_edge(
            u,
            v,
            label=data.get("label", ""),
            color="#888",
            width=6,
            font={"size": 40, "strokeWidth": 3, "strokeColor": "#ffffff"}  # 👈 increase font size here
        )

    ensure_dir(os.path.dirname(output_html))
    net.write_html(output_html, open_browser=False)

    # Inject JS for filtering (show only connected nodes)
    highlight_js = """
    <script type="text/javascript">
    var allNodes;
    var highlightActive = false;
    var hiddenNodes = [];

    network.on("click", function(params) {
        if (params.nodes.length > 0) {
            var selectedNode = params.nodes[0];
            var connectedNodes = network.getConnectedNodes(selectedNode);
            connectedNodes.push(selectedNode);

            allNodes = nodes.get({ returnType: "Object" });
            hiddenNodes = [];

            for (var nodeId in allNodes) {
                if (connectedNodes.indexOf(nodeId) == -1) {
                    hiddenNodes.push(nodeId);
                }
            }

            // Hide all unconnected nodes
            network.body.data.nodes.update(hiddenNodes.map(id => ({ id: id, hidden: true })));

        } else {
            // Reset view
            if (hiddenNodes.length > 0) {
                network.body.data.nodes.update(hiddenNodes.map(id => ({ id: id, hidden: false })));
                hiddenNodes = [];
            }
        }
    });
    </script>
    """

    with open(output_html, "a") as f:
        f.write(highlight_js)

    print(f"Graph visualization saved to: {output_html}")



if __name__ == "__main__":
    G = build_knowledge_graph()
    visualize_graph()
