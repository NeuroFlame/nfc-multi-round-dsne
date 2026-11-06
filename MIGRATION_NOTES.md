# Migration from coinstac-dsne-multishot

This computation was ported from the COINSTAC
[coinstac-dsne-multishot](https://github.com/trendscenter/coinstac-dsne-multishot)
(`local.py`, `remote.py`, `tsneFunctions.py`) to the
[computation-nvflare-boilerplate](https://github.com/NeuroFlame/computation-nvflare-boilerplate)
at commit `ea2436e` (boilerplate 0.1.0, NVFlare 2.8.0).

## Phase mapping

The COINSTAC version selected a function based on the previous side's
`computation_phase` string. Here, an `iterative_workflow` repeats one
local/remote pair, and a `phase` field on each message selects the behavior:

| COINSTAC | NeuroFLAME round | Function |
| --- | --- | --- |
| `local_noop` | 0, `setup` | `compute_site_round` reports data shape |
| `remote_1` | 0 | `aggregate_site_round` embeds the reference data |
| `local_1` | 1, `initialize` | `compute_site_round` builds combined affinities |
| `remote_2` | 1 | `aggregate_site_round` starts optimization |
| `local_2` | 2+, `optimize` | `compute_site_round` takes one gradient step |
| `remote_3` | 2+ | `aggregate_site_round` averages shared updates |
| `local_3` + `remote_4` | final | `build_outputs` writes the plot and embedding at each site |

`iterative_workflow` is used instead of `stepped_workflow` because the number
of rounds comes from the `max_iterations` parameter at run time. The
framework's iteration cap is fixed when the spec is imported, so it is set to
the supported maximum (5000 + 2 rounds). `stop_when=is_complete` ends the run
when central has assembled the final embedding.

Sites echo the phase and iteration they just processed. Central rejects a
round in which sites disagree.

## Data movement changes

- **Reference data.** The original baked `remote_data/` into the image at
  `/computation/remote_data`. It now lives at
  `app/code/computation/reference_data/`. Central sends the normalized
  reference matrix to sites through the framework's `artifact()` transfer, so
  a larger reference set does not hit the 8 MiB inline array limit.
- **Site cache.** The original saved `local_Y`, `local_dY`, `local_IY`,
  `local_P` and `local_gains` as `.npy` files in the COINSTAC cache directory
  on every iteration. They now live in the framework `cache_dir`. The
  affinity matrix `P` is rewritten only when it changes (once, when early
  exaggeration ends).
- **Outputs.** The original produced `final_embed_value.npy` and
  `sample_fig.png` on central only. Every site now writes
  `global_embedding.csv`, `tsne_embedding.png`, `summary.json` and
  `index.html`. The HTML report (`report.py`) uses the shared NeuroFLAME
  report design from the sibling computations (`nfc-combatdc`, `nfc-dpsvm`,
  and others): header chips, site pills, section navigation, cards, stat
  tables, a light/dark theme toggle and inline SVG charts.
- **Site identity.** Each site mints a random token in its input step and
  sends it in the setup round. Central maps tokens to display names and
  returns the map with the final embedding, so a site can name itself in its
  report. This is the same pattern as the sibling computations.

## Numerical behavior

The port first reproduced the original math exactly (see Validation). It
then fixed three bugs in the original that kept site points from converging:

- **Site gains are now persisted.** `local_2` received the updated adaptive
  gains from `master_child` but re-saved the incoming array, so site gains
  stayed at 1 for the whole run. `local_math._optimize` now saves the updated
  gains.
- **Early exaggeration is fully removed.** The original multiplied `P` by 8
  and divided by 4 at iteration 100, leaving 2x exaggeration for the rest of
  the run. The port uses the standard factor of 4 and divides by the same
  factor. (Factor 8 with a full /8 was also tested; 4 scored slightly better
  and matches reference t-SNE.)
- **PCA keeps the leading components.** The original used `np.linalg.eig`
  and took the first `initial_dims` columns without sorting by eigenvalue.
  `pca()` now uses `np.linalg.eigh` (the scatter matrix is symmetric) and
  sorts by descending eigenvalue.

Other deliberate differences:

- **3-D embeddings.** The original hard-coded two dimensions in site
  initialization, centering and plotting. `no_dims` now accepts 2 or 3.
  Initialization draws one column per dimension in the original order, so
  seeded 2-D results are unchanged (verified bit-identical against the
  previous release). The 3-D report view is a dependency-free rotatable SVG,
  so the report stays one self-contained file.

- **Vectorized gradient.** The per-point Python gradient loop was replaced by
  the equivalent matrix expression. It matches the original `master_child`
  to about 1e-14.
- **Iteration numbering.** The original used COINSTAC's global iteration
  counter, which was already at about 3 when optimization began, and ran
  about `max_iterations - 3` gradient steps. This port numbers gradient steps
  from 0 and runs exactly `max_iterations`. The momentum switch (iteration 20),
  the end of early exaggeration (100) and the cost cadence (every 10) apply to
  that 0-based counter.
- **Randomness.** The original reseeded from OS entropy (`np.random.seed()`)
  before every random draw, so runs were not reproducible. The optional
  `random_seed` parameter now seeds both central and site initialization.
  Leaving it unset keeps the original non-reproducible behavior.
- **Validation.** Mismatched feature counts, label/data length mismatches,
  non-finite values, `no_dims != 2`, and perplexity too large for the
  reference set now fail with clear errors.

Kept from the original, by design rather than as bugs:

- Site initialization draws every combined row uniformly within the
  reference embedding's range. The shared rows are replaced with the central
  embedding at the first optimize step.
- The gradient omits the conventional factor of 4, which only rescales the
  learning rate. This is the same as the reference `tsne.py` this code
  derives from.

## Runtime workarounds

Two framework/runtime interactions needed computation-side handling:

- **BLAS pinned to one thread.** Inside the NVFlare simulator, central hung
  indefinitely in `np.linalg.eig` (reference PCA) at ~0% CPU. The same call
  completes in under a second outside NVFlare. This matches the known
  multithreaded-OpenBLAS deadlock in processes forked from a multithreaded
  parent. `tsne.py` calls
  `threadpoolctl.threadpool_limits(1, user_api="blas")` at import. The
  matrices here are small, so the speed cost is negligible.
- **`GlobalMessage.shared_X` is typed `Any`.** The framework replaces the
  received artifact tag with an `ArtifactRef` *before* typed deserialization,
  and the typed path then rejects it ("Cannot deserialize ArtifactRef as
  dataclass ArtifactRef"). Leaving the field untyped takes the untyped path,
  which passes the `ArtifactRef` through. This is probably worth fixing in
  the boilerplate's `serialization.py`, which should accept a value that is
  already an instance of the expected type.

## Validation

- `make check`: lint, format, compile, the 133 boilerplate tests, and 14
  computation tests in `tests/test_dsne_computation.py` all pass.
- NVFlare simulator, 3 sites (`./run_local_simulation.sh site1,site2,site3`):
  completes and writes identical `global_embedding.csv` at every site, with
  80 reference + 40/20/20 site points. Each iteration is one federated round,
  about 10 s per round in the simulator on an Apple Silicon host (amd64
  emulation).
- **Port fidelity, before the fixes.** In-process over 1000 iterations,
  compared with a transcription of the original COINSTAC loop on the same
  data: 5-NN label agreement 0.64 vs 0.65, final KL 7.03 vs 7.70.
- **Effect of the fixes.** In-process, 1000 iterations, mean ± sd over 5
  seeds, 5-NN majority label agreement:

  | Version | Site points | Reference points | Overall |
  | --- | --- | --- | --- |
  | Faithful port (original behavior) | 0.62 ± 0.17 | 0.92 | 0.77 ± 0.13 |
  | Fixed, exaggeration x8 then /8 | 0.91 ± 0.06 | 0.97 | 0.94 ± 0.04 |
  | Fixed, exaggeration x4 then /4 (shipped) | 0.96 ± 0.03 | 0.97 | 0.97 ± 0.02 |

  Final KL error for seed 0 dropped from about 7 to 0.47.

## Boilerplate tests

`tests/test_public_api.py` was adjusted in two places. The two example-spec
assertions now check this computation's task names and local state type. The
expected public API set now includes `cache_file`, which the boilerplate's
`framework/__init__.py` exports but its own test at `ea2436e` did not list.

## Open questions

### Should inputs move to `covariates.csv` / `data.csv`, and how should the reference set be supplied?

**Status:** deferred (2026-11-05). Inputs remain `site_data.txt` and
`site_label.txt`, with the bundled MNIST reference set.

**Proposal.** Switch to the `covariates.csv` / `data.csv` convention used by
the sibling FreeSurfer computations:

- `data.csv`: the features to embed, with a header row and the columns chosen
  in the settings, like the siblings' `Dependents`.
- `covariates.csv`: the columns to color the map by (for example diagnosis,
  sex or age group). The report would let viewers switch between them,
  replacing the single label file.

**The file format alone does not make typical datasets usable.** Two other
changes are needed:

1. **A matching reference set.** dSNE works only through a reference dataset
   that every site can see: central embeds it and sites anchor their points
   to it. The reference must have the same columns as the sites' data,
   measured the same way. The bundled reference (80 MNIST digit images, 784
   pixel features) fits only MNIST-style digit images, so, for example, a
   consortium embedding 17 FreeSurfer volumes needs a reference of those same
   17 volumes from subjects whose data is not private.
2. **Per-feature scaling.** `normalize_rows` divides each row by its largest
   absolute value, which suits pixel data but not tabular features with
   different units. In the sibling FreeSurfer test data, intracranial volume
   (about 1.5 million) would swamp every other column. Each column should be
   standardized with means and standard deviations pooled across sites; sites
   would share only per-column sums and sums of squares.

**Options for supplying the reference set:**

- **A. Consortium-provided reference (recommended).** The consortium leader
  supplies a reference file, such as public data (ABIDE, OASIS, or
  consented subjects) processed with the same pipeline as the sites' data.
  The leader's site sends it to central and it is shared with all sites.
  This follows the `consortium_leader_id` pattern of the brain-age
  computations, and works for any tabular data type without code changes per
  data type.
- **B. Curated reference sets bundled per data type** (for example FreeSurfer
  aseg volumes). Simpler for users, but each set must be curated and
  maintained, and processing versions must match the sites' data.
- **C. A synthetic reference built from pooled statistics.** Needs no real
  reference data, but departs from the published algorithm and would need its
  own validation. Not recommended without a research decision.

**Expected scope with A plus pooled scaling.** Tabular data such as FreeSurfer
ROI tables and FNC vectors would work. Voxel-level VBM (100,000+ columns)
would need downsampling first: the reference matrix is shared with every
site, and PCA builds a features × features matrix.

**If adopted, also update:** `inputs.py` (CSV loading and column selection),
`tsne.normalize_rows` (pooled per-feature standardization, which adds one
round), `remote_math` (accepting the reference from the leader's site), the
report's color-by options, `display_notes.md`, the README and the test data.
