# Decentralized Multi-Shot t-SNE (dSNE-MS) for NeuroFLAME

A NeuroFLAME computation that builds a joint t-SNE embedding of data held at
several sites without moving any site's raw data. It is a port of
[coinstac-dsne-multishot](https://github.com/trendscenter/coinstac-dsne-multishot)
to the
[computation-nvflare-boilerplate](https://github.com/NeuroFlame/computation-nvflare-boilerplate)
framework (NVFlare 2.8, Python 3.11).

## How it works

1. **Reference embedding (central).** Central embeds a shared reference
   dataset bundled with the image
   (`app/code/computation/reference_data/`) using standard t-SNE.
2. **Initialization (sites).** Central sends the normalized reference data
   (as a file artifact) and its embedding to every site. Each site stacks the
   reference data on top of its own data and computes the combined affinity
   matrix locally.
3. **Optimization (rounds).** Each round, every site takes one gradient step
   on its combined data. It moves only its own points and reports its proposed
   update for the shared (reference) points. Central averages those updates
   into the next shared embedding and returns it, along with the global mean
   used for centering.
4. **Output.** After `max_iterations` rounds, sites send their final local
   coordinates and labels. Central assembles the combined embedding and every
   site writes it out.

Only low-dimensional coordinates of the reference points, per-site means, and
the cost value cross the network during optimization. Site coordinates and
labels are shared once, at the end, to build the combined plot (as in the
original COINSTAC computation).

## Site inputs

Each site's data directory contains:

| File | Contents |
| --- | --- |
| `site_data.txt` | One sample per row, one feature per column. Whitespace-delimited, or comma-delimited for `.csv`. No header. The feature count must match the reference data (784 for the bundled MNIST reference). |
| `site_label.txt` | One numeric label per row, in the same order as `site_data.txt`. |

The filenames can be changed with the `site_data` and `site_label`
parameters.

## Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `site_data` | string | `site_data.txt` | Site data filename. |
| `site_label` | string | `site_label.txt` | Site label filename. |
| `no_dims` | int | `2` | Embedding dimensions. Only `2` is supported. |
| `initial_dims` | int | `30` | PCA dimensions computed before t-SNE. |
| `perplexity` | number | `30` | t-SNE perplexity. Must be smaller than the number of reference samples. |
| `max_iterations` | int | `1000` | Decentralized gradient iterations (1 to 5000). Each iteration is one federated round. |
| `random_seed` | int or null | none | Seed for reproducible initialization. Omit for a fresh random start each run. |
| `log_level` | string | `info` | Framework log level. |

Example (`test_data/server/parameters.json`):

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

## Outputs

Every site receives:

| File | Contents |
| --- | --- |
| `global_embedding.csv` | `x`, `y`, `label`, and `source` (`reference` or the site name) for every point. |
| `tsne_embedding.png` | Scatter plot of the combined embedding, colored by label. |
| `summary.json` | Final cost, sample counts per site, and the parameters used. |
| `index.html` | A report with the plot and summary embedded. |

## Running locally

```bash
./run_local_simulation.sh site1,site2,site3
```

Results are written to `test_output/simulate_job/<site>/`. The bundled test
data has three sites with MNIST digits 0 and 9 (40, 20, and 20 samples).

Each iteration is one NVFlare round, so a 1000-iteration run takes many
rounds. Lower `max_iterations` for quick checks.

Run lint, format checks, and tests with:

```bash
make check
```

## Code layout

Author code lives in `app/code/computation/`:

| File | Purpose |
| --- | --- |
| `spec.py` | Declares the iterative workflow. |
| `types.py` | Payload, state, and phase definitions. |
| `inputs.py` | Site and reference data loading. |
| `tsne.py` | t-SNE math ported from the original `tsneFunctions.py`. |
| `local_math.py` | Site rounds (ported from `local.py`). |
| `remote_math.py` | Central rounds (ported from `remote.py`). |
| `results.py` | Final output files. |

`app/code/framework/`, `app/code/runtime/`, and the NVFlare and provisioning
files are boilerplate-managed. Update them with the boilerplate's
`scripts/migrate_computation.py`, not by hand.

See [MIGRATION_NOTES.md](MIGRATION_NOTES.md) for how the COINSTAC phases map
to this workflow, and for behavior carried over from the original.
