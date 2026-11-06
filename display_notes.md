**Source code:** [https://github.com/NeuroFlame/nfc-multi-round-dsne](https://github.com/NeuroFlame/nfc-multi-round-dsne)

&nbsp;

# Decentralized Multi-Shot t-SNE (dSNE-MS)

### Overview

This computation builds a joint t-SNE visualization of data held at several sites, without any site sharing its raw data. It implements the multi-shot decentralized t-SNE algorithm described in:

> Saha DK, Calhoun VD, Panta SR, and Plis SM. *See without looking: joint visualization of sensitive multi-site datasets.* Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI), 2017.

t-SNE places each sample as a point in a 2-D (or 3-D) map so that samples with similar features end up near each other. It is used to see whether samples group by label, spot outliers, or check for site effects across a consortium.

The sites are linked through a shared **reference dataset** that every participant can see. The central node embeds the reference data first. Each site then embeds its own samples together with the reference samples, which act as common anchors. Over many rounds, the sites' proposed updates for the reference points are averaged centrally, so every site's points are placed in one common map.

**Iterations and run time:** every optimization iteration is one federated round, and a run takes `max_iterations` + 2 rounds. With the default of 1000 iterations, expect a long run, as each round needs every site to respond.

### Example Settings

```json
{
  "site_data": "site_data.txt",
  "site_label": "site_label.txt",
  "no_dims": 2,
  "initial_dims": 30,
  "perplexity": 30,
  "max_iterations": 1000,
  "random_seed": 0
}
```

### Settings Specification

| Variable Name | Type | Description | Allowed Options | Default | Required |
| --- | --- | --- | --- | --- | --- |
| `site_data` | `string` | Name of each site's data file. | a filename | `"site_data.txt"` | ❌ false |
| `site_label` | `string` | Name of each site's label file. | a filename | `"site_label.txt"` | ❌ false |
| `no_dims` | `int` | Dimensions of the embedding. `2` gives the standard flat map; `3` gives a map the report lets you rotate. | `2`, `3` | `2` | ❌ false |
| `initial_dims` | `int` | Number of principal components kept (PCA) before t-SNE runs. | at least `no_dims`, at most the number of features | `30` | ❌ false |
| `perplexity` | `number` | Roughly the number of close neighbors each point considers. Larger values emphasize broader structure. | greater than `0` and smaller than the number of reference samples (80 for the bundled reference set) | `30` | ❌ false |
| `max_iterations` | `int` | Number of optimization iterations. Each iteration is one federated round. | `1` – `5000` | `1000` | ❌ false |
| `random_seed` | `int` | Seed for the random starting positions. Set it to make a run reproducible; leave it out for a different random start each run. | any integer | not set | ❌ false |
| `log_level` | `string` | Detail level of the computation logs. | `"debug"`, `"info"`, `"warning"`, `"error"`, `"critical"` | `"info"` | ❌ false |

Every site uses the same settings, so all sites must name their files as `site_data` and `site_label` specify.

### Input Description

Each site provides two plain-text files in its data directory:

1. **Data file** (`site_data.txt` by default)
2. **Label file** (`site_label.txt` by default)

**Data File**

- **Format**: numeric text with no header row. Values are separated by spaces or tabs, or by commas if the filename ends in `.csv`.
- **Rows**: one sample per row.
- **Columns**: one feature per column. The number of columns must match the reference dataset (784 for the bundled reference set).
- **Values**: all values must be present and finite (no blanks, `NaN`, or infinity). Each row is scaled by its largest absolute value before embedding.

**Label File**

- **Format**: numeric text, one label per row, no header.
- **Rows**: must have the same number of rows as the data file, in the same order.
- **Values**: numeric labels, such as `0` and `9`. Labels are used only to color and assess the map; they do not affect where points are placed. Whole-number labels with up to 20 distinct values are treated as categories in the report.

**General Structure**:

`site_data.txt` (shortened)

```
0	0	0	12	255	...	0
0	0	34	188	201	...	0
```

`site_label.txt`

```
0
9
```

### Algorithm Description

The key steps of the algorithm include:

1. **Setup**:
    - Each site loads its data and labels, checks them, and reports its sample count and feature count to the central node.
    - The central node checks that every site's feature count matches the reference dataset, and that the settings are valid.

2. **Reference embedding (central node)**:
    - The central node scales the reference data, reduces it to `initial_dims` principal components, and runs standard t-SNE for 1000 iterations to embed it.
    - It sends the scaled reference data and its embedding to every site.

3. **Site initialization**:
    - Each site stacks the reference samples on top of its own samples and computes pairwise similarities for the combined set, using `perplexity`.
    - Each site draws random starting positions for its own points within the area covered by the reference embedding.

4. **Decentralized optimization (one round per iteration)**:
    - Each site takes one t-SNE gradient step on its combined data. It moves only its own points, and reports its proposed update for the reference points, its mean position, and its current error.
    - The central node averages the sites' proposals into the next reference embedding, recenters the map, and sends the result back to every site.
    - Early exaggeration is used for the first 100 iterations to help clusters form, then removed.

5. **Final assembly**:
    - In the last iteration, each site also sends the final positions and labels of its own points.
    - The central node combines the reference points and every site's points into one embedding and sends it to all sites, and each site writes its outputs.

**What leaves a site:** during optimization, only positions of the shared reference points, a mean position, and an error value. At the end, each site's final low-dimensional positions and labels are shared with all participating sites to build the combined map. Raw site data never leaves its site.

### Assumptions

- **The reference dataset defines the data format.** The bundled reference set is 80 MNIST handwritten-digit images (digits 0 and 9), each flattened to 784 pixel values. Every site's data must have the same number of features, and represent the same kind of measurement, as the reference set. To use other data (for example, neuroimaging features), the reference set bundled with the computation must be replaced.
- `perplexity` must be smaller than the number of reference samples.
- Each site's data and label files have the same number of rows, in the same order, and contain only numeric values.
- All sites use the same file names, since settings are shared across the consortium.
- Distances between clusters and the axis values of the map carry no meaning in t-SNE; only which points are near each other does.
- t-SNE results depend on the random start. Use `random_seed` when results must be reproduced.

### Output Description

- **Output files**: `global_embedding.csv`, `tsne_embedding.png`, `summary.json`, and an `index.html` report at every site.
- All sites receive the same combined embedding; the report names and highlights the site viewing it.

**Combined Embedding** (`global_embedding.csv`):

| Column | Description |
| --- | --- |
| `point` | Row number in the combined embedding |
| `x`, `y` | Position in the map |
| `z` | Third coordinate, for 3-D runs only |
| `label` | The point's label |
| `source` | `reference`, or the name of the site the point belongs to |

**Run Summary** (`summary.json`):

| Field | Description |
| --- | --- |
| `site` | Name of the site that wrote this file |
| `final_error` | Final Kullback–Leibler (KL) error, averaged over sites. Lower means the map preserves neighborhoods better. |
| `n_reference_samples` | Number of reference points |
| `n_site_samples` | Number of points from each site |
| `n_local_samples` | Number of points from this site |
| `kl_error_history` | `[iteration, KL error]` pairs, recorded every 10 iterations |
| `parameters` | The settings used for the run |

**Plot** (`tsne_embedding.png`): a static image of the combined embedding colored by label. For 3-D runs it shows one fixed viewing angle.

**Results Report** (`index.html`):

- **Embedding**: an interactive map of every point. Hover a point for its source and label, and color points by label or by source (this site, reference, other sites). With more than three labels, points are colored by source and an extra **Embedding by label** section shows one small panel per label. For 3-D runs, drag the map or use the arrow keys to rotate it.
- **Convergence**: KL error over the iterations after early exaggeration ends.
- **Neighbor label agreement**: for each site and the reference set, the share of points whose label matches most of their 5 nearest neighbors in the map. This describes how cleanly labels cluster; it is not a statistical test.
- **Run settings** and download links for the output files.
