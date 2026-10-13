# Continuous empirical genotype-error scaling

## Goal

Replace the binary empirical genotype-error toggle with a continuous multiplier and standardise the names of all three error parameters.

Use these names everywhere in Python, Snakemake wildcards, configuration keys, function arguments, result dataframe columns, and tests:

```text
geno_multiplier
phase_ser
mispol_rate
```

The only exception is **generated filenames/directories**, where the labels should remain concise:

```text
geno-{geno_multiplier}-phase{phase_ser}-mispol{mispol_rate}
```

For example:

```text
...-geno-2.0-phase0.001-mispol0.01-...
```

Keep this change narrowly scoped. Do not refactor unrelated pipeline code.

Automated tests must remain only in:

```text
tests/test_errors.py
```

and should remain basic, direct tests like the existing suite.

Do not create pipeline/Snakemake integration tests.

---

# 1. Configuration interface

Change each error profile from the current form:

```yaml
error_configs:
  - empgeno: "off"
    phase: 0
    mispol: 0
    name: "none"
```

to:

```yaml
error_configs:
  - geno_multiplier: 0.0
    phase_ser: 0.0
    mispol_rate: 0.0
    name: "none"
```

Interpretation:

```text
geno_multiplier = 0.0   no genotype errors
geno_multiplier = 1.0   original empirical matrix
geno_multiplier = 2.0   twice the empirical error probability where possible
```

`geno_multiplier` must be a finite float >= 0.

`phase_ser` and `mispol_rate` remain ordinary probabilities and must remain in:

```text
0 <= value <= 1
```

If the ignored local `config.yaml` exists, migrate its profiles in the same way while preserving all other local configuration.

For old profiles:

```text
empgeno: "off" -> geno_multiplier: 0.0
empgeno: "on"  -> geno_multiplier: 1.0
```

Do not add backwards compatibility for the old config keys.

---

# 2. Explicit genotype-error scaling algorithm

Implement the scaling at the **3x3 diploid-genotype matrix level**, before the current conversion to the ordered 4x4 genotype representation.

The empirical matrix for one allele-frequency bin is:

```text
        observed genotype
          0     1     2

true 0   p00   p01   p02
true 1   p10   p11   p12
true 2   p20   p21   p22
```

For true genotype `i`:

- `p_ii` is the probability of a correct genotype call;
- the other two entries are genotype-error probabilities.

Add a small helper with a clear interface such as:

```python
scale_genotype_error_probs(probs, geno_multiplier)
```

where `probs` is the 3x3 matrix.

## Algorithm for each row

For row `i`, let the two off-diagonal probabilities be:

```text
e1, e2
```

First multiply them:

```text
scaled_e1 = e1 * geno_multiplier
scaled_e2 = e2 * geno_multiplier
```

and calculate:

```text
off_diag_sum = scaled_e1 + scaled_e2
```

### Case A: `off_diag_sum < 1`

Keep the multiplied off-diagonal probabilities unchanged:

```text
new_e1 = scaled_e1
new_e2 = scaled_e2
```

and rebalance the correct-call probability:

```text
new_diagonal = 1 - off_diag_sum
```

Therefore, for nonsaturated rows:

```text
new error probability
    = original error probability * geno_multiplier
```

### Case B: `off_diag_sum >= 1`

The multiplied values can no longer form a valid probability row.

Calculate the normalization constant:

```text
normalization_constant = 1 / off_diag_sum
```

Then multiply **both already-scaled off-diagonal entries by the same constant**:

```text
new_e1 = scaled_e1 * normalization_constant
new_e2 = scaled_e2 * normalization_constant
```

and set:

```text
new_diagonal = 0
```

The resulting off-diagonal probabilities therefore sum exactly to 1.

This preserves their empirical relative proportions:

```text
new_e1 / new_e2 = e1 / e2
```

whenever both are nonzero.

For example, if a row has:

```text
correct = 0.40
error A = 0.50
error B = 0.10
```

then with:

```text
geno_multiplier = 2
```

the initially multiplied errors are:

```text
1.00
0.20
```

with:

```text
off_diag_sum = 1.20
normalization_constant = 1 / 1.20
```

so the final row becomes approximately:

```text
correct = 0
error A = 0.833333
error B = 0.166667
```

The original 5:1 ratio between the two error classes is retained.

Do **not** clip individual probabilities independently.

Do **not** redistribute excess probability back onto the diagonal.

Do **not** perform any dataset-wide adjustment.

The transformation is independent for every true-genotype row.

## Required invariants

After scaling every row:

```text
all probabilities >= 0
all probabilities <= 1
row sum == 1
```

within normal floating-point tolerance.

Special cases follow naturally:

### `geno_multiplier = 0`

All off-diagonal probabilities become zero and each diagonal becomes 1:

```text
identity matrix
```

### `geno_multiplier = 1`

The original empirical probability matrix is recovered, apart from insignificant floating-point normalization for any already-saturated row.

### Very large multiplier

Each row with any nonzero empirical error probability eventually becomes:

```text
diagonal = 0
off-diagonal probabilities summing to 1
```

while retaining the empirical relative proportions of its error classes.

---

# 3. Refactor `fetch_empirical_probs`

Change its signature to:

```python
fetch_empirical_probs(freq, df, geno_multiplier=1.0)
```

Keep the existing allele-frequency lookup unchanged.

After selecting the empirical CSV row, first construct the explicit 3x3 matrix:

```python
[
    [row.p00, row.p01, row.p02],
    [row.p10, row.p11, row.p12],
    [row.p20, row.p21, row.p22],
]
```

Pass this matrix to:

```python
scale_genotype_error_probs(...)
```

Then convert the scaled 3x3 result into the existing 4x4 ordered representation.

Keep that expansion semantically unchanged:

```text
00
01
10
11
```

In particular:

- genotype `0 -> 1` probability remains split equally between `01` and `10`;
- genotype `2 -> 1` probability remains split equally between `01` and `10`;
- the two heterozygous input orientations retain the current implementation's behaviour.

Do not change the empirical frequency lookup/binning logic.

---

# 4. Pass `geno_multiplier` through genotype-error generation

Keep:

```python
add_empirical_genotype_errors(...)
```

structurally simple.

Its existing `probs_func(..., **kwargs)` approach is sufficient.

Pass:

```python
geno_multiplier=geno_multiplier
```

through to `fetch_empirical_probs`.

Do not introduce another error-model abstraction.

---

# 5. Update `add_errors`

Change the function interface to use:

```python
add_errors(
    ds,
    output_path,
    error_csv_path,
    geno_multiplier,
    phase_ser,
    mispol_rate,
    seed,
)
```

Remove:

```text
empirical_errors_enabled
phase_error_rate
mispol_error_rate
```

Validate:

```text
geno_multiplier >= 0
0 <= phase_ser <= 1
0 <= mispol_rate <= 1
```

## `geno_multiplier == 0`

Preserve the efficient no-error path:

```python
G_geno_error = G_in
```

Do not read the empirical CSV and do not consume RNG values for genotype errors.

This preserves the existing RNG behaviour for configurations with no genotype error.

## `geno_multiplier > 0`

Load the empirical matrix and call:

```python
add_empirical_genotype_errors(...)
```

with the specified multiplier.

Keep unchanged:

- genotype error mask generation;
- genotype error count generation;
- phase-switch ordering;
- mispolarisation ordering;
- output dataset construction.

---

# 6. Rename phase-error arguments consistently

Use:

```text
phase_ser
```

everywhere the phase-switch error rate is represented.

For example, change:

```python
sample_phase_switches(d_in, ser, rng)
```

to:

```python
sample_phase_switches(d_in, phase_ser, rng)
```

and:

```python
add_phase_switch_errors(G_in, switch_error_rate, rng)
```

to:

```python
add_phase_switch_errors(G_in, phase_ser, rng)
```

Inside these functions use `phase_ser` as well.

The meaning remains:

```text
probability of a phase switch per adjacent heterozygous-site pair
```

Do not change the phase-switch algorithm.

---

# 7. Use `mispol_rate` consistently

`unbiased_mispolarise` already uses the desired:

```python
mispol_rate
```

name.

Propagate this name consistently through callers.

Replace remaining names such as:

```text
mispol_error_rate
mispol
```

when they represent a Python/config parameter.

Do not change the mispolarisation algorithm.

---

# 8. Update the Snakefile names

Use the full parameter names in all Python and wildcard code:

```text
geno_multiplier
phase_ser
mispol_rate
```

Start by finding every old occurrence:

```bash
rg -n \
    'empgeno|empirical_errors_enabled|phase_error_rate|switch_error_rate|\bser\b|mispol_error_rate' \
    Snakefile lib tests config.yaml.example
```

Also inspect generic `phase` and `mispol` names in Python/config code and replace them when they refer to these error parameters.

## `map_error_profiles`

Change the interface to:

```python
map_error_profiles(geno_multiplier, phase_ser, mispol_rate)
```

Read:

```python
err_conf["geno_multiplier"]
err_conf["phase_ser"]
err_conf["mispol_rate"]
```

Convert them to floats.

Use `math.isclose` for all three comparisons.

Keep returning the configured profile `name`.

---

# 9. Filename convention

Although full names are used in code, generated filenames should use only:

```text
geno
phase
mispol
```

as labels.

Every affected path should follow:

```text
-geno-{geno_multiplier}-phase{phase_ser}-mispol{mispol_rate}
```

For example:

```text
...-rep0-geno-2.0-phase0.001-mispol0.01.zarr
```

and:

```text
...-rep0-geno-2.0-phase0.001-mispol0.01-v0.5.1-ancestors.zarr
```

The **wildcard names themselves remain the full names**:

```text
{geno_multiplier}
{phase_ser}
{mispol_rate}
```

Only the human-readable filename labels are abbreviated.

Apply this consistently to:

- error-modified Zarr paths;
- ancestor outputs;
- ancestor-generation logs;
- chunk directories;
- processed chunk paths;
- aggregated per-replicate dataframes;
- any other generated path currently containing the old error parameters.

No compatibility with old `empgeno-*` output paths is required.

---

# 10. Update `rule add_errors`

Read:

```python
geno_multiplier = float(wildcards.geno_multiplier)
phase_ser = float(wildcards.phase_ser)
mispol_rate = float(wildcards.mispol_rate)
```

Validate their permitted ranges.

Call:

```python
errors.add_errors(
    ...,
    geno_multiplier=geno_multiplier,
    phase_ser=phase_ser,
    mispol_rate=mispol_rate,
    ...
)
```

Do not retain any `"on"` / `"off"` genotype-error logic.

---

# 11. Propagate full wildcard names through the workflow

Rename local variables and wildcard arguments throughout the Snakefile.

Affected areas include at least:

```text
zarr_with_errors
generate_ancestors
expand_ancestors_by_version
build_ancestor_chunks
chunk_input
process_ancestor_chunk
get_checkpoint
process_all_chunks
aggregate_ancestor_chunks
aggregated_dataframe_paths
map_error_profiles
```

Use:

```python
wildcards.geno_multiplier
wildcards.phase_ser
wildcards.mispol_rate
```

where appropriate.

Do not leave abbreviated Python variables such as:

```text
geno
phase
mispol
```

Those abbreviations are reserved for filename labels only.

---

# 12. Rename result-dataframe fields

In `lib/utils.py`, change the `process_ancestor_chunk` interface to:

```text
geno_multiplier
phase_ser
mispol_rate
```

and write these exact dataframe columns:

```text
geno_multiplier
phase_ser
mispol_rate
```

Do not use:

```text
geno
phase
mispol
```

as dataframe field names.

The short forms are for filenames only.

Pass numeric values into the dataframe.

Do not change any other result columns.

---

# 13. Tests: keep them only in `tests/test_errors.py`

Do not create additional test files.

Keep the tests small and deterministic.

Existing unrelated tests for encoding, phase switching and mispolarisation should remain.

Adjust their argument names as required by the API rename, for example:

```python
phase_ser=0
```

rather than:

```python
switch_error_rate=0
```

and:

```python
phase_ser=1
```

rather than:

```python
ser=1
```

## Add basic multiplier tests

Add a small class such as:

```text
TestGenotypeErrorMultiplier
```

Do not create extensive statistical simulations.

### Test 1: multiplier zero

Using a simple synthetic 3x3 probability matrix:

```text
geno_multiplier = 0
```

must produce the identity matrix.

### Test 2: multiplier one

Using the same matrix:

```text
geno_multiplier = 1
```

must reproduce it.

Use `np.allclose`.

### Test 3: ordinary doubling

Use a row whose original off-diagonal total is comfortably below 0.5.

For example:

```text
[0.90, 0.08, 0.02]
```

With:

```text
geno_multiplier = 2
```

it should become:

```text
[0.80, 0.16, 0.04]
```

Test corresponding rows where the diagonal is in positions 1 and 2 as well, either in the same small matrix or with simple assertions.

### Test 4: saturation preserves relative error proportions

Use an intentionally unbalanced row such as:

```text
[0.40, 0.50, 0.10]
```

for a true genotype whose diagonal is the first entry.

With:

```text
geno_multiplier = 2
```

the initial scaled off-diagonal probabilities are:

```text
1.0
0.2
```

so:

```text
off_diag_sum = 1.2
normalization_constant = 1 / 1.2
```

Expected result:

```text
[0.0, 5/6, 1/6]
```

Assert:

- diagonal is zero;
- off-diagonal probabilities sum to one;
- their original 5:1 ratio is retained.

This directly tests the intended saturation algorithm.

### Test 5: exact boundary

Include a basic case where:

```text
off_diag_sum * geno_multiplier == 1
```

and verify that:

```text
diagonal == 0
```

and the off-diagonal probabilities sum to 1.

This explicitly covers the required `>= 1` branch.

### Test 6: negative multiplier

Verify that a negative:

```text
geno_multiplier
```

is rejected.

Do not build a larger validation framework.

---

# 14. Adjust existing empirical-matrix tests

Keep the existing basic `fetch_empirical_probs` tests.

Where they call:

```python
fetch_empirical_probs(...)
```

either rely on the default:

```text
geno_multiplier=1.0
```

or specify `geno_multiplier=1.0` when doing so makes the test clearer.

Keep the existing test that rows sum to one.

Keep the extreme-frequency test if it still describes valid behaviour at multiplier 1.

Do not add a statistical test requiring observed random error counts to be exactly twice as large at multiplier 2.

The deterministic probability-matrix tests are the correct place to verify the multiplier.

---

# 15. Do not alter unrelated behaviour

Do not change:

- the empirical CSV;
- allele-frequency lookup;
- frequency binning;
- genotype encoding;
- genotype sampling;
- phase-switch mechanics;
- mispolarisation mechanics;
- RNG seeding;
- tsinfer selection;
- simulations;
- ancestor comparison;
- resource settings;
- aggregation logic other than parameter/column renaming.

Do not introduce interpolation between empirical frequency bins.

Do not add a new error-model class or abstraction.

---

# 16. Validation

Run only the existing error test suite:

```bash
uv run pytest tests/test_errors.py
```

Then:

```bash
uv run ruff check .
```

Do not add other automated tests.

## Check old terminology

Run:

```bash
rg -n \
    'empgeno|empirical_errors_enabled|phase_error_rate|switch_error_rate|\bser\b|mispol_error_rate' \
    Snakefile lib tests config.yaml.example
```

There should be no remaining uses representing these parameters.

Names such as:

```text
add_empirical_genotype_errors
fetch_empirical_probs
```

should remain unchanged: they accurately describe the empirical model.

## Check abbreviated names

Inspect uses of:

```bash
rg -n '\bgeno\b|\bphase\b|\bmispol\b' \
    Snakefile lib tests config.yaml.example
```

Any abbreviation referring to an error parameter in Python/config/dataframe code should be replaced by its full name.

Abbreviations are expected only as literal filename components such as:

```text
geno-{geno_multiplier}-phase{phase_ser}-mispol{mispol_rate}
```

Do not blindly replace unrelated uses of the word `phase` that describe actual haplotype phase state inside the phase-switch algorithm.

## Check new terminology

Run:

```bash
rg -n \
    'geno_multiplier|phase_ser|mispol_rate' \
    Snakefile lib tests config.yaml.example
```

Confirm all three parameters propagate consistently from config through the workflow and into output dataframe metadata.

---

# 17. Smoke test

After the error unit tests pass, run the same small laptop Snakemake configuration that is currently known to complete successfully.

Use a simple profile such as:

```yaml
geno_multiplier: 0.0
phase_ser: 0.0
mispol_rate: 0.0
name: "none"
```

If practical, also run a genotype-error profile with:

```yaml
geno_multiplier: 1.0
```

to exercise the empirical CSV path.

This is a manual smoke test only.

Do not add it to the automated test suite.

---

# Completion criteria

The work is complete when:

1. `empgeno` has been replaced by `geno_multiplier`;
2. phase-switch error parameters consistently use `phase_ser`;
3. mispolarisation parameters consistently use `mispol_rate`;
4. config uses all three full names;
5. Python functions and Snakefile variables use all three full names;
6. result dataframe columns use all three full names;
7. filenames alone use the short labels:

   ```text
   geno-{geno_multiplier}-phase{phase_ser}-mispol{mispol_rate}
   ```

8. `geno_multiplier=0` produces no genotype error;
9. `geno_multiplier=1` reproduces the empirical matrix;
10. nonsaturated rows scale off-diagonal probability linearly;
11. rows with multiplied off-diagonal mass `>= 1` are renormalized over their off-diagonal entries and have diagonal probability zero;
12. saturation preserves the relative proportions of the two empirical error classes;
13. only `tests/test_errors.py` has automated tests;
14. the error tests pass;
15. ruff passes;
16. the small pipeline smoke test still completes.

At completion report:

- files changed;
- final config syntax;
- exact matrix-scaling implementation;
- functions whose parameter names changed;
- all filename-pattern changes;
- tests added/updated;
- pytest result;
- ruff result;
- smoke-test result;
- `git status --short`.

Do not commit unless explicitly asked.