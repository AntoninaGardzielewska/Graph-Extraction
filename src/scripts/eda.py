from src.utils.config_loader import load_config_yaml
from pathlib import Path

config = load_config_yaml("base.yaml")

data_folder = Path(config["data"]["path_to_data"])

print(data_folder)

# %%
from src.utils.config_loader import load_config_yaml
from pathlib import Path

config = load_config_yaml("base.yaml")

data_folder = Path(config["data"]["path_to_data"])

print(data_folder)

# %%
import re
from bisect import bisect_right
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 160)

# %% [markdown]
# ## 1. File discovery

# %%
# Adjust if the corpus lives in a subfolder of data_folder, e.g. data_folder / "RegulaTome"
print("Configured data folder:", data_folder.resolve())
print("Data folder exists:", data_folder.exists())

DATA_ROOT = data_folder

print("RegulaTome folder:", DATA_ROOT.resolve())
print("RegulaTome folder exists:", DATA_ROOT.exists())

ann_files = sorted(DATA_ROOT.rglob("*.ann"))
txt_files = sorted(DATA_ROOT.rglob("*.txt"))

print("Annotation files:", len(ann_files))
print("Text files:", len(txt_files))

if DATA_ROOT.exists():
    print("\nContents:")
    for path in DATA_ROOT.iterdir():
        print(path.name)

if not ann_files:
    raise FileNotFoundError(
        "No .ann files found. Check DATA_ROOT and the extracted dataset."
    )
print(f".ann files: {len(ann_files)}")
print(f".txt files: {len(txt_files)}")

# Split name = name of the folder directly containing the file (relative to root)
def get_split(path: Path) -> str:
    rel = path.relative_to(DATA_ROOT)
    return rel.parts[0] if len(rel.parts) > 1 else "root"

split_counts = Counter(get_split(p) for p in ann_files)
print("Files per split:", dict(split_counts))

# Check that every .ann has a matching .txt
missing_txt = [p for p in ann_files if not p.with_suffix(".txt").exists()]
print(f".ann without matching .txt: {len(missing_txt)}")

# %% [markdown]
# ## 2. BRAT parser

# %%
def parse_ann(ann_path: Path):
    """Parse a BRAT .ann file into entities, relations, events, attributes, normalizations."""
    entities, relations, events, attributes, norms = [], [], [], [], []

    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            idx = parts[0]
            kind = idx[0]

            try:
                if kind == "T":
                    type_and_spans = parts[1]
                    text = parts[2] if len(parts) > 2 else ""
                    etype, spans = type_and_spans.split(" ", 1)
                    span_list = [tuple(map(int, s.split())) for s in spans.split(";")]
                    entities.append({
                        "ent_id": idx,
                        "type": etype,
                        "start": span_list[0][0],
                        "end": span_list[-1][1],
                        "n_fragments": len(span_list),
                        "text": text,
                    })
                elif kind == "R":
                    fields = parts[1].split()
                    rtype = fields[0]
                    args = dict(a.split(":", 1) for a in fields[1:])
                    relations.append({"rel_id": idx, "type": rtype, **args})
                elif kind == "E":
                    events.append({"event_id": idx, "raw": parts[1]})
                elif kind == "A" or kind == "M":
                    fields = parts[1].split()
                    attributes.append({
                        "attr_id": idx,
                        "name": fields[0],
                        "target": fields[1] if len(fields) > 1 else None,
                        "value": fields[2] if len(fields) > 2 else None,
                    })
                elif kind == "N":
                    norms.append({"norm_id": idx, "raw": parts[1], "text": parts[2] if len(parts) > 2 else ""})
            except Exception as e:  # keep going on odd lines, but report
                print(f"[WARN] {ann_path.name}: could not parse line: {line!r} ({e})")

    return entities, relations, events, attributes, norms


# %%
doc_rows, ent_rows, rel_rows, evt_rows, attr_rows, norm_rows = [], [], [], [], [], []

for ann_path in ann_files:
    doc_id = ann_path.stem
    split = get_split(ann_path)
    txt_path = ann_path.with_suffix(".txt")
    text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""

    ents, rels, evts, attrs, norms = parse_ann(ann_path)

    doc_rows.append({
        "doc_id": doc_id,
        "split": split,
        "n_chars": len(text),
        "n_words": len(text.split()),
        "n_entities": len(ents),
        "n_relations": len(rels),
        "n_events": len(evts),
        "n_attributes": len(attrs),
        "text": text,
    })
    for e in ents:
        ent_rows.append({"doc_id": doc_id, "split": split, **e})
    for r in rels:
        rel_rows.append({"doc_id": doc_id, "split": split, **r})
    for a in attrs:
        attr_rows.append({"doc_id": doc_id, "split": split, **a})
    for n in norms:
        norm_rows.append({"doc_id": doc_id, "split": split, **n})
    for ev in evts:
        evt_rows.append({"doc_id": doc_id, "split": split, **ev})

docs = pd.DataFrame(doc_rows)
ents = pd.DataFrame(ent_rows)
rels = pd.DataFrame(rel_rows)
attrs = pd.DataFrame(attr_rows)
norms = pd.DataFrame(norm_rows)
events = pd.DataFrame(evt_rows)

print(f"docs={len(docs)}  entities={len(ents)}  relations={len(rels)}  "
      f"events={len(events)}  attributes={len(attrs)}  normalizations={len(norms)}")

# %% [markdown]
# ## 3. Overview per split

# %%
overview = docs.groupby("split").agg(
    n_docs=("doc_id", "count"),
    chars_mean=("n_chars", "mean"),
    words_mean=("n_words", "mean"),
    entities=("n_entities", "sum"),
    relations=("n_relations", "sum"),
    docs_without_relations=("n_relations", lambda s: (s == 0).sum()),
    docs_without_entities=("n_entities", lambda s: (s == 0).sum()),
).round(1)
overview.loc["TOTAL"] = [
    len(docs), docs.n_chars.mean().round(1), docs.n_words.mean().round(1),
    docs.n_entities.sum(), docs.n_relations.sum(),
    (docs.n_relations == 0).sum(), (docs.n_entities == 0).sum(),
]
overview

# %% [markdown]
# ## 4. Document length

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(data=docs, x="n_words", hue="split", bins=60, element="step", ax=axes[0])
axes[0].set_title("Document length (words)")
sns.boxplot(data=docs, x="split", y="n_words", ax=axes[1])
axes[1].set_title("Document length by split")
plt.tight_layout()
plt.show()

docs[["n_chars", "n_words"]].describe(percentiles=[.05, .25, .5, .75, .95, .99]).round(1)

# %% [markdown]
# ## 5. Entities

# %%
ents["length_chars"] = ents["end"] - ents["start"]
ents["length_words"] = ents["text"].str.split().str.len()

ent_type_counts = ents.groupby(["split", "type"]).size().unstack(0, fill_value=0)
ent_type_counts["total"] = ent_type_counts.sum(axis=1)
ent_type_counts.sort_values("total", ascending=False)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
order = ents["type"].value_counts().index
sns.countplot(data=ents, y="type", hue="split", order=order, ax=axes[0])
axes[0].set_title("Entity types per split")
sns.histplot(data=ents, x="length_words", bins=range(1, 15), ax=axes[1])
axes[1].set_title("Entity mention length (words)")
plt.tight_layout()
plt.show()

# %%
# Mentions per document
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(data=docs, x="n_entities", bins=60, ax=ax)
ax.set_title("Entities per document")
plt.show()

# Discontinuous mentions
print("Discontinuous entities:", (ents["n_fragments"] > 1).sum(),
      f"({(ents['n_fragments'] > 1).mean():.2%})")

# %%
# Most frequent surface forms (lowercased), per type
ents["text_norm"] = ents["text"].str.lower().str.strip()
top_surface = (ents.groupby(["type", "text_norm"]).size()
               .rename("count").reset_index()
               .sort_values("count", ascending=False))
top_surface.groupby("type").head(10).reset_index(drop=True)

# %%
# Lexical diversity: unique surface forms vs. mentions
div = ents.groupby("type").agg(mentions=("text_norm", "size"), unique_forms=("text_norm", "nunique"))
div["unique_ratio"] = (div["unique_forms"] / div["mentions"]).round(3)
div

# %%
# Ambiguity: surface forms that appear under more than one entity type
amb = ents.groupby("text_norm")["type"].nunique()
print(f"Surface forms with >1 entity type: {(amb > 1).sum()} / {len(amb)}")
amb[amb > 1].sort_values(ascending=False).head(15)

# %% [markdown]
# ## 6. Relations

# %%
print("Relation columns:", list(rels.columns))
arg_cols = [c for c in rels.columns if c.lower().startswith("arg")]
print("Argument columns:", arg_cols)

rel_type_counts = rels.groupby(["split", "type"]).size().unstack(0, fill_value=0)
rel_type_counts["total"] = rel_type_counts.sum(axis=1)
rel_type_counts.sort_values("total", ascending=False)

# %%
fig, ax = plt.subplots(figsize=(9, 4))
order = rels["type"].value_counts().index
sns.countplot(data=rels, y="type", hue="split", order=order, ax=ax)
ax.set_title("Relation types per split")
plt.tight_layout()
plt.show()

# %%
# Relations per document
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(data=docs, x="n_relations", bins=60, ax=axes[0])
axes[0].set_title("Relations per document")
axes[1].scatter(docs["n_words"], docs["n_relations"], s=8, alpha=0.4)
axes[1].set_xlabel("words"); axes[1].set_ylabel("relations")
axes[1].set_title("Document length vs. relations")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.1 Join relations with entity info (types, positions, distances)

# %%
if len(arg_cols) >= 2 and len(rels):
    a1, a2 = arg_cols[0], arg_cols[1]
    ent_lookup = ents.set_index(["doc_id", "ent_id"])[["type", "start", "end", "text"]]

    def join_arg(df, col, suffix):
        keys = pd.MultiIndex.from_arrays([df["doc_id"], df[col]])
        sub = ent_lookup.reindex(keys).reset_index(drop=True)
        sub.columns = [f"{c}_{suffix}" for c in sub.columns]
        return pd.concat([df.reset_index(drop=True), sub], axis=1)

    rj = join_arg(rels, a1, "a1")
    rj = join_arg(rj, a2, "a2")

    unresolved = rj["type_a1"].isna() | rj["type_a2"].isna()
    print(f"Relations with unresolved arguments (e.g. event-type args): {unresolved.sum()}")
    rj = rj[~unresolved].copy()

    rj["pair"] = rj["type_a1"] + " -> " + rj["type_a2"]
    rj["char_distance"] = (rj[["start_a1", "start_a2"]].max(axis=1)
                           - rj[["end_a1", "end_a2"]].min(axis=1)).clip(lower=0)
else:
    rj = pd.DataFrame()
    print("Could not identify two argument columns; skipping joined analysis.")

# %%
if len(rj):
    # Relation type x argument entity-type pair
    display(pd.crosstab(rj["type"], rj["pair"]))

    # Directionality: how often is the same unordered pair annotated in both directions
    rj["ordered"] = list(zip(rj["doc_id"], rj["text_a1"].str.lower(), rj["text_a2"].str.lower()))
    ordered_set = set(rj["ordered"])
    reverse_hits = sum((d, b, a) in ordered_set for d, a, b in ordered_set)
    print(f"Ordered pairs with reverse pair also annotated: {reverse_hits} / {len(ordered_set)}")

    # Self-relations
    print("Self-relations (same arg id):", (rj[a1] == rj[a2]).sum())

# %%
if len(rj):
    # Sentence-level vs cross-sentence relations
    sent_bounds = {}
    for doc_id, text in zip(docs["doc_id"], docs["text"]):
        starts = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+", text)]
        sent_bounds[doc_id] = starts

    def sent_idx(doc_id, pos):
        return bisect_right(sent_bounds[doc_id], pos) - 1

    rj["sent_a1"] = [sent_idx(d, p) for d, p in zip(rj["doc_id"], rj["start_a1"])]
    rj["sent_a2"] = [sent_idx(d, p) for d, p in zip(rj["doc_id"], rj["start_a2"])]
    rj["sent_distance"] = (rj["sent_a1"] - rj["sent_a2"]).abs()
    rj["cross_sentence"] = rj["sent_distance"] > 0

    print(f"Cross-sentence relations: {rj['cross_sentence'].mean():.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(data=rj, x="char_distance", bins=60, ax=axes[0])
    axes[0].set_title("Character distance between arguments")
    sns.countplot(data=rj, x=rj["sent_distance"].clip(upper=10), ax=axes[1])
    axes[1].set_title("Sentence distance between arguments (10 = 10+)")
    plt.tight_layout()
    plt.show()

    display(rj.groupby("type")[["char_distance", "sent_distance"]].describe().round(1))

# %% [markdown]
# ## 7. Attributes, normalizations, events

# %%
if len(attrs):
    print("Attribute names:")
    display(attrs.groupby(["name", "value"], dropna=False).size().rename("count").reset_index())
else:
    print("No attribute (A/M) lines found.")

if len(norms):
    print(f"Normalization lines: {len(norms)}")
    display(norms.head())
else:
    print("No normalization (N) lines found.")

if len(events):
    print(f"Event lines: {len(events)}")
    display(events.head())

# %% [markdown]
# ## 8. Data quality checks

# %%
# 8.1 Entity span text matches the document text?
doc_text = docs.set_index("doc_id")["text"].to_dict()
bad = []
for row in ents[ents["n_fragments"] == 1].itertuples():
    snippet = doc_text[row.doc_id][row.start:row.end]
    if snippet != row.text:
        bad.append((row.doc_id, row.ent_id, row.text, snippet))
print(f"Entity spans that don't match document text: {len(bad)}")
if bad:
    print(bad[:5])

# 8.2 Duplicate entities (same span & type in a doc)
dups = ents.duplicated(subset=["doc_id", "start", "end", "type"], keep=False).sum()
print(f"Duplicate entity spans: {dups}")

# 8.3 Duplicate relations
if len(rj):
    dup_rel = rj.duplicated(subset=["doc_id", "type", a1, a2], keep=False).sum()
    print(f"Duplicate relations: {dup_rel}")

# 8.4 Documents with no annotations
print(f"Docs with no entities: {(docs.n_entities == 0).sum()}")
print(f"Docs with no relations: {(docs.n_relations == 0).sum()}")

# %% [markdown]
# ## 9. Class balance summary and sample

# %%
if len(rels):
    balance = rels["type"].value_counts(normalize=True).mul(100).round(2).rename("percent")
    display(balance)

# Peek at a random annotated document
sample = docs[docs.n_relations > 0].sample(1, random_state=42).iloc[0]
print(f"doc_id={sample.doc_id}  split={sample.split}  entities={sample.n_entities}  relations={sample.n_relations}")
print(sample.text[:1500])
display(ents[ents.doc_id == sample.doc_id][["ent_id", "type", "start", "end", "text"]].head(20))
display(rels[rels.doc_id == sample.doc_id].head(20))