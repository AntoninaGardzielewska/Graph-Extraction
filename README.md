# Automatic knowledge graph extraction from text

Large amounts of information are stored in unstructured text, making it difficult to organize, search, and analyze. 
Knowledge graphs provide a structured representation of information by capturing entities and relationships between them.
The aim of this thesis is to develop and compare methods for automatic knowledge graph extraction from text. 
The methods will be evaluated using publicly available datasets. The specific approaches will be selected during 
the course of the work depending on data availability and computational resources and the primary use case will be
in biomedical domain.


## Setup
 
```bash
pip install -r requirements.txt
pip install -e .
```
 
1. Download RegulaTome from Zenodo (DOI `10.5281/zenodo.10808330`) into `data/`.
2. Set the paths in `resources/base.yaml` (see [`resources/README.md`](resources/README.md)).
## Run
 
```bash
python -m src.generate_graphs
```
 
Prints a graph summary and saves an interactive HTML graph to `visualization.output_path`.
 