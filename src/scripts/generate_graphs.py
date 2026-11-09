import logging
from pathlib import Path

import pandas as pd, networkx as nx

from src.utils.config_loader import load_config_yaml
from src.utils.logging_config import configure_logging

_logger = logging.getLogger(__name__)

def parse_ann(path):
    ents, rels = {}, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("T"):
            tid, meta, text = line.split("\t")
            etype = meta.split()[0]
            ents[tid] = {"type": etype, "text": text}
        elif line.startswith("R"):
            _, meta = line.split("\t")[:2]
            rtype, a1, a2 = meta.split()
            rels.append((rtype, a1.split(":")[1], a2.split(":")[1]))
    return ents, rels

def get_triplets(folder_path):
    rows = []
    for ann in Path(folder_path).rglob("*.ann"):
        ents, rels = parse_ann(ann)
        for rtype, a1, a2 in rels:
            rows.append({
                "subject": ents[a1]["text"].lower(), "subject_type": ents[a1]["type"],
                "predicate": rtype,
                "object": ents[a2]["text"].lower(), "object_type": ents[a2]["type"],
                "doc": ann.stem,
            })

if __name__ == "__main__":
    configure_logging()

    config = load_config_yaml()