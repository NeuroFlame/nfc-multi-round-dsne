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
  `index.html`.

## Numerical behavior

The math is kept equivalent to the original, with these deliberate
differences:

- **Vectorized gradient.** The per-point Python gradient loop was replaced by
  the equivalent matrix expression. On the bundled data it matches the
  original `master_child` to about 1e-14.
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

## Behavior carried over from the original

These are kept as-is so results stay comparable. Each one is worth reviewing
on the algorithm side:

1. **Site gains are never updated.** `local_2` received updated gains from
   `master_child` but re-saved the incoming array, so the cached site gains
   stay at 1. The port keeps this (see the comment in
   `local_math._optimize`).
2. **Exaggeration is never fully removed.** `P` is multiplied by 8 and later
   divided by 4, so a 2x exaggeration remains for the rest of the run.
   Reference t-SNE uses x4 then /4.
3. **PCA eigenvectors are unsorted.** `pca()` uses `np.linalg.eig` and takes
   the first `initial_dims` columns without sorting by eigenvalue.
4. **Site initialization ignores shared positions.** Every combined row is
   drawn uniformly within the reference embedding's range. The shared rows are
   replaced with the central embedding at the first optimize step.
5. **No factor of 4 in the gradient**, matching the original implementation.

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

- `make check`: lint, format, compile and all 133 boilerplate tests pass.
- NVFlare simulator, 3 sites (`./run_local_simulation.sh site1,site2,site3`):
  completes and writes identical `global_embedding.csv` at every site, with
  80 reference + 40/20/20 site points. Each iteration is one federated round,
  about 10 s per round in the simulator on an Apple Silicon host (amd64
  emulation).
- In-process comparison over 1000 iterations against a transcription of the
  original COINSTAC loop, on the same data. 5-NN label agreement is 0.64 for
  the port vs 0.65 for the original (reference points 0.82 vs 0.81), and the
  final KL is 7.03 vs 7.70. In both, site points separate poorly on this small
  test set. That is a property of the original algorithm (see the carried-over
  behavior above), not of the port.

## Boilerplate tests

`tests/test_public_api.py` was adjusted in two places. The two example-spec
assertions now check this computation's task names and local state type. The
expected public API set now includes `cache_file`, which the boilerplate's
`framework/__init__.py` exports but its own test at `ea2436e` did not list.
