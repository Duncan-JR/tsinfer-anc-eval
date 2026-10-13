# Set up Jupytext notebooks and initial error analysis

## Goal

Add a lightweight notebook workflow to `tsinfer-anc-eval` that:

- uses the existing uv project environment;
- keeps notebook-specific dependencies in a separate uv dependency group;
- uses Jupytext paired notebooks;
- creates `notebooks/error_analysis.ipynb`;
- pairs it with `notebooks/error_analysis.py` using Jupytext's `py:percent` format;
- makes the `.py` representation the preferred file for agents to edit;
- includes an initial analysis of the genotype-error Zarr datasets generated for `geno_multiplier = 0, 1, 2, 5`;
- plots genome-wide genotype-matrix cell error rate as a function of `geno_multiplier`.

Do not refactor pipeline code as part of this task.

Do not add notebook tests to pytest.

Do not alter the error-generation implementation.

---

## 1. Add a dedicated uv notebook dependency group

The repo already uses uv dependency groups and currently has a `lint` group.

Add a separate group called:

```text
notebooks
```

using uv rather than manually editing package versions.

From the repository root run:

```bash
uv add --group notebooks jupyterlab jupytext ipykernel matplotlib nbconvert
```

The existing core dependencies already provide packages needed by this notebook, including:

```text
numpy
pandas
sgkit
xarray
zarr
```

Do not duplicate those into the notebook group.

Do not add notebook dependencies to `[project.dependencies]`.

Do not add `notebooks` to `tool.uv.default-groups`; notebook tooling should remain optional.

Afterwards verify:

```bash
uv sync --group notebooks
```

Expected repository changes:

```text
pyproject.toml
uv.lock
```

---

## 2. Add root Jupytext configuration

Create:

```text
jupytext.toml
```

at the repository root.

Use:

```toml
formats = "ipynb,py:percent"
```

Keep the configuration minimal.

This means notebooks are paired as:

```text
notebooks/error_analysis.ipynb
notebooks/error_analysis.py
```

The `.ipynb` is used normally in JupyterLab and retains notebook outputs.

The `.py` file is the text representation that agents should normally read and edit.

Do not add pre-commit hooks, nbstripout, notebook execution frameworks, or other notebook infrastructure yet.

---

## 3. Create the notebooks directory

Create:

```text
notebooks/
```

The first notebook should be:

```text
notebooks/error_analysis.ipynb
```

paired with:

```text
notebooks/error_analysis.py
```

Use the `py:percent` representation so cells appear in the text file as:

```python
# %% [markdown]
# Markdown cell

# %%
# Python cell
```

This format is deliberately chosen because it is easy for both humans and coding agents to edit.

---

## 4. Create the paired notebook

A simple implementation path is:

1. create `notebooks/error_analysis.py` in `py:percent` format;
2. convert it to an `.ipynb`;
3. mark the two files as a pair;
4. sync them.

For example:

```bash
uv run --group notebooks jupytext \
    --to ipynb notebooks/error_analysis.py
```

then:

```bash
uv run --group notebooks jupytext \
    --set-formats ipynb,py:percent \
    notebooks/error_analysis.ipynb
```

then:

```bash
uv run --group notebooks jupytext \
    --sync notebooks/error_analysis.ipynb
```

After this, both files should exist.

Do not hand-edit the JSON in the `.ipynb`.

---

# 5. Initial notebook structure

Keep this first notebook very small.

It should contain approximately four cells.

## Cell 1 - title and purpose

Markdown:

```text
# Genotype error analysis

Validate the empirical genotype-error scaling by measuring the realised
genotype-matrix cell error rate for datasets generated with different
`geno_multiplier` values.

The error rate used here is:

    number of altered allele cells / total number of genotype-matrix cells

where `call_genotype_error_mask` is False for an altered cell.
```

Do not call this literal "errors per base pair", because the denominator is genotype matrix cells rather than physical sequence length.

"Genome-wide genotype error rate" or "genotype-matrix cell error rate" is appropriate.

---

## Cell 2 - imports and paths

Use only straightforward imports:

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sgkit
```

Determine the repository root without relying on Jupyter being launched from one particular working directory.

Keep this simple, for example:

```python
repo_root = Path.cwd()
if not (repo_root / "pyproject.toml").exists():
    repo_root = repo_root.parent

zarr_dir = repo_root / "data" / "zarr_vcfs"
```

Assert that:

```python
zarr_dir.exists()
```

with a useful error message.

Do not introduce a general repository-path utility.

---

## Cell 3 - load generated datasets and calculate error rates

Analyse exactly the dataset generated from:

```yaml
model: OutOfAfrica_4J17
contig: chr20
samples:
  CEU: 200
  CHB: 200
  YRI: 200
seed: 1
replicate: 0
```

so the dataset has:

```text
n600
s1
rep0
```

and use only profiles with:

```text
phase_ser = 0.0
mispol_rate = 0.0
```

for these multipliers:

```python
geno_multipliers = [0.0, 1.0, 2.0, 5.0]
```

Do not hard-code the entire filename including the formatting of `left` and `right`.

Instead discover the corresponding Zarr directory for each multiplier using the established filename structure, while constraining the glob to the intended simulation.

The filenames currently follow the pipeline convention:

```text
...-geno-{geno_multiplier}-phase{phase_ser}-mispol{mispol_rate}.zarr
```

Use a pattern along the lines of:

```python
pattern = (
    "OutOfAfrica_4J17-chr20-"
    "*-n600-s1-rep0-"
    f"geno-{geno_multiplier}-phase0.0-mispol0.0.zarr"
)
```

For each multiplier:

1. find the corresponding Zarr directory;
2. assert exactly one path matches;
3. load it using:

```python
ds = sgkit.load_dataset(path, consolidated=False)
```

4. select:

```python
error_mask = ds.call_genotype_error_mask
```

The mask semantics are:

```text
True  = allele cell unchanged
False = allele cell changed by genotype-error simulation
```

Calculate:

```text
num_cells = total number of elements in error_mask
num_errors = number of False elements
error_rate = num_errors / num_cells
```

Prefer lazy aggregation rather than converting the full mask to a NumPy array.

For example, use logic equivalent to:

```python
num_cells = error_mask.size
num_correct = int(error_mask.sum().compute().item())
num_errors = num_cells - num_correct
error_rate = num_errors / num_cells
```

Store one row per multiplier with at least:

```text
geno_multiplier
num_errors
num_cells
error_rate
zarr_path
```

Create a dataframe:

```python
error_rates
```

sorted by `geno_multiplier`.

Display it as the final expression of the cell.

Do not calculate errors from inferred ancestors or downstream dataframes; this notebook is directly validating the simulated genotype-error mask.

---

## Cell 4 - visualise realised scaling

Plot:

```text
x = geno_multiplier
y = error_rate
```

Use a simple line plot with markers.

The axes should be clearly labelled:

```text
Genotype error multiplier
Genotype-matrix cell error rate
```

Title:

```text
Genome-wide genotype error rate
```

Also add a useful reference for testing the multiplier behaviour.

Take the realised error rate at:

```text
geno_multiplier = 1.0
```

and calculate:

```python
linear_expectation = rate_at_1x * geno_multiplier
```

Plot this as a dashed reference line alongside the observed rates.

This gives an immediate visual test of whether:

```text
x2 ≈ 2 × x1
x5 ≈ 5 × x1
```

while allowing deviations caused by saturation of high-error empirical matrix rows.

Also add a dataframe column such as:

```text
rate_relative_to_x1
```

defined as:

```text
error_rate / x1_error_rate
```

with the zero-error case set to `0.0`.

The plot does not need publication styling yet.

Keep it deliberately simple and easy to modify.

Example conceptual plotting code:

```python
x1_error_rate = error_rates.loc[
    error_rates["geno_multiplier"] == 1.0, "error_rate"
].iloc[0]

error_rates["linear_expectation"] = (
    error_rates["geno_multiplier"] * x1_error_rate
)
error_rates["rate_relative_to_x1"] = (
    error_rates["error_rate"] / x1_error_rate
)
error_rates.loc[
    error_rates["geno_multiplier"] == 0, "rate_relative_to_x1"
] = 0.0

fig, ax = plt.subplots()

ax.plot(
    error_rates["geno_multiplier"],
    error_rates["error_rate"],
    marker="o",
    label="Observed",
)

ax.plot(
    error_rates["geno_multiplier"],
    error_rates["linear_expectation"],
    linestyle="--",
    label="Linear from 1×",
)

ax.set_xlabel("Genotype error multiplier")
ax.set_ylabel("Genotype-matrix cell error rate")
ax.set_title("Genome-wide genotype error rate")
ax.legend()

fig
```

Follow the repo's ordinary Python formatting.

Do not add a plotting helper module yet.

---

# 6. Important interpretation of the metric

The calculation requested here is:

```text
num erroneous allele cells
--------------------------
total genotype matrix cells
```

For a genotype matrix with shape:

```text
(variants, samples, ploidy)
```

the denominator is:

```text
variants × samples × ploidy
```

For these diploid simulations:

```text
ploidy = 2
```

A diploid genotype in which both alleles change therefore contributes two erroneous cells.

Do not instead count "number of genotypes that changed".

Do not divide by the physical sequence length of 10 Mb.

That would be a different metric and can be added later if wanted.

---

# 7. Version-control policy

Keep both:

```text
notebooks/error_analysis.ipynb
notebooks/error_analysis.py
```

in the repository for now.

Do not add either extension to `.gitignore`.

For future agent work, the preferred source to modify is:

```text
notebooks/error_analysis.py
```

After editing it, run:

```bash
uv run --group notebooks jupytext --sync \
    notebooks/error_analysis.py
```

This should update the paired `.ipynb`.

Do not have an agent edit the `.ipynb` JSON directly.

---

# 8. Validate Jupytext pairing

Run:

```bash
uv run --group notebooks jupytext --sync \
    notebooks/error_analysis.ipynb
```

Then test the round trip:

```bash
uv run --group notebooks jupytext \
    --test notebooks/error_analysis.ipynb \
    --to py:percent
```

Both files should remain synchronized.

---

# 9. Execute the notebook once

Because the generated Zarr data already exists locally, execute the notebook once to verify the analysis actually runs.

Use:

```bash
uv run --group notebooks jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    notebooks/error_analysis.ipynb
```

Run this from the repository root.

Verify that:

- all four multipliers are found;
- the `0.0` dataset has zero genotype errors;
- the dataframe contains four rows;
- all `num_cells` values are the same;
- all rates are finite;
- the plot is produced;
- the observed `1×`, `2×`, and `5×` points increase in the expected direction.

Do not hard-code numerical expected rates yet.

This notebook is exploratory validation, not a unit test.

After execution, synchronize the pair once more if necessary:

```bash
uv run --group notebooks jupytext --sync \
    notebooks/error_analysis.ipynb
```

---

# 10. Existing tests

Do not add notebook-specific pytest tests.

Run the existing basic error tests only as a regression check:

```bash
uv run pytest tests/test_errors.py
```

Also run:

```bash
uv run ruff check .
```

If Ruff attempts to lint notebook-generated content awkwardly, do not introduce broad lint configuration changes without a demonstrated need.

---

# 11. Completion state

Expected new files:

```text
jupytext.toml

notebooks/
├── error_analysis.ipynb
└── error_analysis.py
```

Expected modified files:

```text
pyproject.toml
uv.lock
```

Do not modify:

```text
Snakefile
lib/errors.py
lib/utils.py
config.yaml.example
```

unless an actual notebook setup problem requires it.

At completion report:

1. notebook dependency-group contents;
2. files created;
3. Jupytext pairing format;
4. whether round-trip validation passed;
5. whether the notebook executed successfully;
6. the resulting four error-rate rows;
7. whether `geno_multiplier=0` produced exactly zero errors;
8. observed relative rates for 1×, 2×, and 5×;
9. existing error-test result;
10. Ruff result;
11. `git status --short`.

Do not commit unless explicitly asked.