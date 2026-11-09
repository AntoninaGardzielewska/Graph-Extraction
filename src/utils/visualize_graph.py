import logging
from pathlib import Path

import pandas as pd
from pyvis.network import Network
import networkx as nx

_logger = logging.getLogger(__name__)

def print_graph_summary(G: nx.MultiDiGraph, top_n: int = 10) -> None:
    _logger.info("Nodes: %d | Edges: %d", G.number_of_nodes(), G.number_of_edges())

    rel_counts = pd.Series([d["relation"] for _, _, d in G.edges(data=True)]).value_counts()
    _logger.info("Top relations:\n%s", rel_counts.head(top_n).to_string())

    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:top_n]
    _logger.info("Top nodes by degree:\n%s",
                 "\n".join(f"  {name}: {deg}" for name, deg in degrees))


def select_subgraph(G, center: str | None = None, radius: int = 1, top_n: int = 100):
    if center:
        center = center.lower()
        if center not in G:
            raise ValueError(f"Node '{center}' not in graph")
        return nx.ego_graph(G, center, radius=radius, undirected=True)
    top_nodes = [n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:top_n]]
    return G.subgraph(top_nodes)


def _edge_color(relation: str) -> str:
    rel = relation.lower()
    if "positive" in rel:
        return "green"
    if "negative" in rel:
        return "red"
    return "gray"


def visualize_graph(
    G,
    output_path: str = "graph.html",
    show_edge_labels: bool = True,
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    net = Network(height="900px", width="100%", directed=True, notebook=False, cdn_resources="remote")

    # curved edges so several relations between the same pair don't overlap
    net.set_options("""
    {
      "edges": {
        "font": {"size": 10, "align": "middle"},
        "smooth": {"type": "dynamic"}
      },
      "physics": {"solver": "forceAtlas2Based"}
    }
    """)

    for node, data in G.nodes(data=True):
        net.add_node(node, label=node, title=", ".join(data.get("types", [])))

    for u, v, d in G.edges(data=True):
        label = d["relation"].replace("_", " ") if show_edge_labels else ""
        net.add_edge(
            u, v,
            label=label,
            title=f'{d["relation"]} ({d["support"]} docs)',
            color=_edge_color(d["relation"]),
        )

    net.write_html(str(out))
    _logger.info("Saved graph to %s", out)