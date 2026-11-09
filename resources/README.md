# Configuration

All settings live in `base.yaml`. The file is loaded with `load_config_yaml("base.yaml")`, which looks for
it in this `resources/` folder first, then in the project root. An absolute path also works.

## `data`

| Parameter | Description |
|-----------|-------------|
| `path_to_data` | Root folder with datasets. Used by the EDA notebook. |
| `path_to_ann` | Folder with the RegulaTome `.ann` files (searched recursively). Used to build the graph. |

## `visualization`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `print_summary` | `true` | Log node/edge counts, top relation types and top nodes by degree. |
| `enabled` | `false` | If `true`, save an interactive HTML graph. If `false`, the graph is built but not plotted. |
| `top_n` | `100` | Plot the N most connected nodes. Ignored when `center` is set. |
| `center` | `null` | Node name (lowercase) to plot with its neighborhood, e.g. `"p53"`. The run fails with an error if the node is not in the graph. |
| `radius` | `1` | Number of hops around `center` to include. Only used when `center` is set. |
| `output_path` | `outputs/graph.html` | Where to save the HTML file. Missing folders are created. |
| `show_edge_labels` | `true` | Show relation names on edges. Turn off for dense graphs (the names are still in the hover tooltip). |

## Example

```yaml
data:
  path_to_data: data
  path_to_ann: data/RegulaTome-corpus/graphs

visualization:
  print_summary: true
  enabled: true
  top_n: 100
  center: null
  radius: 1
  output_path: data/plots/graph.html
  show_edge_labels: true
```

## Notes

- Prefer relative paths (as above) over absolute ones, so the config works on other machines.
  Forward slashes work on Windows too. If you use backslashes, don't put the path in double quotes.
- To plot one protein's neighborhood, set `center: "your_protein"` and keep `radius` at 1 or 2.
  Larger radii grow very quickly.
- Edge colors are fixed: green for positive regulation, red for negative regulation, gray for the rest.