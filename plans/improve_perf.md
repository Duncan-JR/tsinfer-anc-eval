# Improve ancestor chunk processing

## Aim

Fix the two largest avoidable costs in `lib/utils.py::process_ancestor_chunk`:
repeated reads from compressed ancestor storage and a large, poorly laid-out
true-genotype matrix. Keep the implementation close to the current row loop.

Preserve the scientific results, CSV columns, and record order. Replace the old
implementation directly, without compatibility paths, fallback algorithms, or
new configuration options. Keep the existing chunk size and Snakemake job
structure. This work should mainly affect `lib/utils.py` and one small test file.

## Why these two changes

1. **Read inferred ancestors through their existing chunk-aware iterator.**
   Individual `ancestor(id)` calls repeatedly decompress storage blocks. On 256
   ancestors from a real chunk, individual reads took about 1.84 seconds versus
   0.30 seconds with `ancestors(indexes)`. This approximately sixfold improvement
   applies to ancestor access, not the whole job.
2. **Store only the true-genotype information actually used.** The current
   `genotype_matrix(...).T` allocates all simulated sites as `int32`; its transpose
   is a view with strided haplotype access. A representative chunk uses 2.27 GiB
   for that matrix. Shared sites encoded as missing/ancestral/derived in a
   contiguous `int8` matrix would occupy about 0.40 GiB. Reducing this working
   set directly addresses the observed memory compression with many workers.

## Storage choice: use what is already available

Use tsinfer's `AncestorData.ancestors(indexes)` for inferred haplotypes. Its
underlying arrays already have chunked storage, and the iterator handles the
block reads. Do not add another Zarr layer or write a custom chunk cache.

Keep decoding true ancestral haplotypes from the tree sequence. The existing
VCF Zarr store represents sampled individuals, whereas this calculation needs
selected internal ancestral nodes and their missingness across the sequence.
Using Zarr for those nodes would require a new materialised dataset, node/site
mappings, and another pipeline stage. That is unnecessary for the current
chunk's compact matrix and is outside this change.

Continue using the existing VCF Zarr dataset for focal-site annotations. Revisit
persisting ancestral genotypes only if later measurements show that repeated
true-genotype decoding remains a substantial bottleneck after these fixes.

## Phase 1: Establish essential correctness tests

Add `tests/test_utils.py`, with class-based tests and small pytest fixtures.
Use explicitly constructed inputs: a tiny tree sequence, small actual
`AncestorData` objects, a dataframe, and an xarray annotation dataset.

Keep two tests:

- **Compact genotype decoding:** compare the helper's output to an explicit
  small expected array. Include an omitted nonshared site, a missing genotype,
  and an allele index greater than one. Check that the entries are correctly
  represented as `-1`, `0`, or `1` in true-node/shared-site order.
- **Complete chunk output:** process a few rows, including repeated inferred
  ancestor IDs and different true nodes. Use differing version boundaries,
  missing true genotypes, and both flipped and unflipped sites. Compare the
  entire parsed CSV, including columns and row order, with a fixed expected
  table. Parameterise this test for one and two versions. Include the terminal
  boundary in the fixture, and make the tiny ancestor store span more than one
  storage block.

Before replacing the current processor, capture its complete expected output
for the small fixture and hand-check representative boundaries and error
counts. Keep those expected tables; do not retain a second implementation as a
reference. The decoder test is added when its helper is introduced in phase 3.

Use direct assertions and existing source-path selection for the configured
tsinfer checkout. Run tests with `uv run`. No performance assertions, mocks of
internal calls, compatibility tests, or additional testing infrastructure.

## Phase 2: Stream inferred ancestors while retaining the row loop

The chunk producer already emits rows in nondecreasing `inf_node` order. Document
that contract; a single monotonicity assertion is enough. Do not sort inputs or
support a second processing path for unsorted data.

1. Obtain the sorted unique inferred IDs as a NumPy integer array.
2. Create one lazy `anc_data.ancestors(indexes=ids)` iterator per version.
3. Keep the current row loop and progress reporting. When `inf_node` changes,
   advance each iterator once and retain the current ancestor for that version.
   Consecutive rows for the same ancestor reuse it.
4. Replace the current ancestors at the next ID. Never materialise all yielded
   ancestors or maintain an additional cache.

Use the existing version-to-ancestor mapping and a current-ID variable. There
is no need for a new preparation class or a grouped outer loop. Preserve the
current order of rows, versions, and left/right records.

While touching this code, replace the two linear boundary scans with
`np.searchsorted(anc_shared_idx, boundary, side="left")`. This is a direct
replacement for counting indices strictly below the boundary. Leave the
remaining boundary and overlap calculations in place.

Run the complete-output test before proceeding.

## Phase 3: Replace the full true-genotype matrix

Add one small helper that returns the compact matrix. Keep the existing unique
true-node list and node-to-row mapping in the caller; no result dataclass or
additional preparation framework is needed for a single returned array.

1. Keep the current expanded-tree-sequence construction and node flags. Changing
   sample/isolation semantics is unrelated to these two performance fixes.
2. Create one `tskit.Variant` decoder with the selected true nodes and explicit
   `isolated_as_missing=True`. Decode the sorted site IDs in `true_shared_idx`
   directly instead of calling `genotype_matrix` for all sites.
3. Fill a shared-site-by-node `int8` array. Classify decoded values as missing
   (`tskit.MISSING_DATA`), ancestral (`0`), or derived (`1`). Convert every
   positive allele index to one before narrowing its dtype.
4. Transpose into a C-contiguous node-by-shared-site array and release the
   staging array. Return that matrix. The two compact arrays briefly coexist;
   the 0.40 GiB example is retained matrix size, not total peak worker memory.
5. In the row loop, access the prepared shared-site row directly. Remove the
   old full-site projection. Keep the existing per-row presence detection,
   boundary calculation, and XOR transformation.

Keeping those existing per-row operations avoids adding boundary caches or
precomputed transformed haplotypes. The important fixes are the matrix's size
and layout. Further work needs evidence that these scans remain significant.

Preserve the following details exactly:

- Find the first and last nonmissing sites before applying XOR. Missing values
  inside those bounds retain the current `(genotype > 0)` treatment.
- Use the existing shared-site mapping of
  `~ds.variant_mispolarisation_mask` for XOR.
- Intersect the true interval with all versions' inferred intervals, using
  half-open bounds. Keep `ts.sites_position[-1] + 1` as the terminal coordinate.
- Leave inferred missing values, directional-error arithmetic, denominators,
  annotations, and CSV writing unchanged.

Run both correctness tests and remove the obsolete matrix code.

## Finish with one practical check

Run one representative full chunk before and after the changes, using the same
inputs and separate output files. Compare all CSV values and record order, and
record elapsed time and peak resident memory. This is a manual check, not a
benchmark suite or a test threshold. Add another chunk only if a discrepancy or
an unresolved bottleneck needs investigation.

Stop once correctness is established and the two costs have been removed.
Keep Snakemake's existing process-level parallelism and one thread per chunk.
Choose future worker counts from measured memory use; do not add automatic
worker tuning, nested multiprocessing, persistent genotype stores, pairwise
caches, or a broader rewrite of the comparison loop.

## Implementation and practical check — 2026-10-13

Completed all three phases in `lib/utils.py` and `tests/test_utils.py`. The fixed
CSV tables were captured from the original processor before either optimisation
and hand-checked for the terminal coordinate, version overlap, and error counts
with missing genotypes. They passed after the iterator change and again after
compact decoding. The decoder test checks explicit values, node/site order,
`int8` dtype, and C-contiguous layout. The full suite passed: 38 tests. Ruff lint,
format checks for the changed Python files, and `git diff --check` passed.

The manual before/after run used the entire 5,000-row `unprocessed-chunk-0.csv`
under:

```text
data/anc_eval2/chunks/OutOfAfrica_4J17-chr20-L0-R1e7-n600-s1-rep0-geno-2.0-phase0.0-mispol0.0/
```

Both runs used its matching simulated tree sequence (124,851 sites; 1,939
selected true nodes), VCF Zarr annotations, and ancestor stores for versions
0.4.1 and 0.5.1, loaded through the first configured tsinfer source checkout.
Parameters were replicate 0, error profile `geno2x`, genotype multiplier 2.0,
phase SER 0.0, and mispolarisation rate 0.0. The runs used separate fresh Python
processes through `uv run`, with the same inputs and one thread per chunk.

| Measurement | Original | Updated |
| --- | ---: | ---: |
| Processor elapsed time | 37.913 s | 7.063 s |
| Peak resident memory | 1,529,348,096 bytes (1.424 GiB) | 1,083,064,320 bytes (1.009 GiB) |

Elapsed time was measured with `time.perf_counter()` around
`process_ancestor_chunk`; peak resident memory was measured with
`resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` for the whole Python process
(bytes on this macOS machine, including imports and input loading). This single
chunk ran 5.37 times faster and used 29.2% less peak resident memory. These are
manual observations, without a performance threshold or benchmark suite.

Outputs were written separately to
`/private/tmp/ancestor_perf_R46J6Z/before.csv` and
`/private/tmp/ancestor_perf_R46J6Z/after.csv`. All 20,000 records and 31 columns
matched exactly with `pandas.testing.assert_frame_equal(check_exact=True)`,
including row order; the CSV files were also byte-identical. The temporary
directory contains the runner, logs, and raw timing/memory measurements for
review. No additional chunk was needed.
