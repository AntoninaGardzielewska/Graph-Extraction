import logging
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

from utils.visualize_graph import print_graph_summary, select_subgraph, visualize_graph
from src.utils.config_loader import load_config_yaml
from src.utils.logging_config import configure_logging

_logger = logging.getLogger(__name__)


def parse_ann(path: Path):
    ents, rels = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("T"):
            tid, meta, text = line.split("\t")
            ents[tid] = {"type": meta.split()[0], "text": text}
        elif line.startswith("R"):
            _, meta = line.split("\t")[:2]
            rtype, a1, a2 = meta.split()
            rels.append((rtype, a1.split(":")[1], a2.split(":")[1]))
    return ents, rels


def get_triplets(folder_path: str) -> pd.DataFrame:
    rows = []
    for ann in Path(folder_path).rglob("*.ann"):
        ents, rels = parse_ann(ann)
        for rtype, a1, a2 in rels:
            if a1 not in ents or a2 not in ents:
                _logger.warning("Missing entity in %s: %s / %s", ann.name, a1, a2)
                continue
            rows.append({
                "subject": ents[a1]["text"].lower(), "subject_type": ents[a1]["type"],
                "predicate": rtype,
                "object": ents[a2]["text"].lower(), "object_type": ents[a2]["type"],
                "doc": ann.stem,
            })
    _logger.info("Extracted %d triples", len(rows))
    return pd.DataFrame(rows)


def build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    # one edge per (subject, predicate, object), with doc support aggregated
    grouped = (
        df.groupby(["subject", "predicate", "object"])["doc"]
        .agg(lambda s: sorted(set(s)))
        .reset_index()
    )
    G = nx.MultiDiGraph()

    node_types = defaultdict(set)
    for r in df.itertuples():
        node_types[r.subject].add(r.subject_type)
        node_types[r.object].add(r.object_type)
    for name, types in node_types.items():
        G.add_node(name, types=sorted(types))   # keeps conflicts visible

    for r in grouped.itertuples():
        G.add_edge(r.subject, r.object, key=r.predicate,
                   relation=r.predicate, docs=r.doc, support=len(r.doc))
    return G


def main(config: dict) -> nx.MultiDiGraph:
    df = get_triplets(config["data"]["path_to_ann"])
    G = build_graph(df)
    _logger.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    viz_cfg = config.get("visualization", {})
    if viz_cfg.get("print_summary", True):
        print_graph_summary(G)
    if viz_cfg.get("enabled", False):
        sub = select_subgraph(
            G,
            center=viz_cfg.get("center"),
            radius=viz_cfg.get("radius", 1),
            top_n=viz_cfg.get("top_n", 100),
        )
        visualize_graph(
            sub,
            output_path=viz_cfg.get("output_path", "outputs/graph.html"),
            show_edge_labels=viz_cfg.get("show_edge_labels", True),
        )
    return G


if __name__ == "__main__":
    configure_logging()
    config = load_config_yaml("base.yaml")
    main(config)