# PART 2 — Use configurable tsinfer installations in the pipeline

## Preconditions

Assume Part 1 has already been completed and reviewed.

The following directories should already exist and work:

```text
~/bin/tsinfer_0_5_1
~/bin/tsinfer_0_4_1
```

Do not redo the cloning or compilation unless verification shows that Part 1 was not actually completed.

Do not modify dependency configuration in this part unless absolutely required by the changes below.

Do not commit anything.

## Goal

Replace the current three-part tsinfer configuration system:

```text
tsinfer_dir
versions
hard-coded paths inside generate_ancestors.py
```

with one list of version/path pairs.

Also remove the obsolete `env_dir` / manual venv activation mechanism now that this repository is run with uv.

The pipeline must naturally support either one configured tsinfer release or several.

Keep downstream comparison and aggregation behaviour unchanged.

## 1. Config schema

The intended configuration structure is:

```yaml
tsinfer:
  - version: "0.5.1"
    path: "~/bin/tsinfer_0_5_1"
  - version: "0.4.1"
    path: "~/bin/tsinfer_0_4_1"
```

Version values must remain strings.

Remove the old:

```yaml
tsinfer_dir: ...
env_dir: ...

versions:
  ...
```

However, **before editing `config.yaml.example`, inspect its current contents**. I may have manually tweaked the example between Part 1 and Part 2.

Preserve all unrelated changes I have made.

Only make the changes necessary to introduce the new `tsinfer` structure and remove the superseded fields.

If the ignored local `config.yaml` exists, migrate its tsinfer-related fields to the same structure while preserving every other local setting.

Never stage or commit `config.yaml`.

## 2. Derive configured releases once in the Snakefile

After the config has been loaded, derive the entries, version list and version-to-path map.

Use simple code along these lines:

```python
tsinfer_entries = config["tsinfer"]

tsinfer_versions = [
    entry["version"]
    for entry in tsinfer_entries
]

tsinfer_paths = {
    entry["version"]: Path(entry["path"]).expanduser()
    for entry in tsinfer_entries
}
```

Do not add a config class, dataclass or new module.

Do not over-abstract this.

## 3. Select the tsinfer used by the Snakefile

The Snakefile itself only needs one tsinfer implementation for loading `AncestorData`.

Use the **first configured tsinfer entry**.

Replace the existing:

```python
sys.path.append(config["tsinfer_dir"])
import tsinfer
```

with logic equivalent to:

```python
snakefile_tsinfer_path = tsinfer_paths[tsinfer_versions[0]]
sys.path.insert(0, str(snakefile_tsinfer_path))

import tsinfer
```

The import must occur after inserting the configured source directory.

Use `insert(0, ...)`, not `append(...)`, so the explicitly configured source checkout takes precedence over any other tsinfer that happens to be importable.

Do not install tsinfer into the project environment.

## 4. Replace every use of the old version list

Remove every use of:

```python
config["versions"]
```

and use:

```python
tsinfer_versions
```

instead.

There are three known places to update.

### `expand_ancestors_by_version`

Continue producing exactly the same naming pattern:

```text
...-v{version}-ancestors.zarr
```

but enumerate over:

```python
tsinfer_versions
```

### `build_ancestor_chunks`

Change the assertion comparing the number of loaded ancestor datasets to use:

```python
len(tsinfer_versions)
```

Do not otherwise change ancestor loading.

### `process_ancestor_chunk`

Use the common:

```python
tsinfer_versions
```

when constructing the existing version-to-`AncestorData` mapping.

Keep:

```text
version -> AncestorData
```

as the conceptual structure.

Do not alter the downstream dataframe or aggregation logic.

## 5. Pass each release's source path to `generate_ancestors`

The existing `{version}` wildcard should remain because versions are still encoded in filenames and downstream results.

In the `generate_ancestors` rule, add:

```python
params:
    tsinfer_path=lambda wildcards: tsinfer_paths[wildcards.version],
```

Keep existing input, output, log, resource and thread behaviour unchanged.

Replace:

```text
--version {wildcards.version}
```

with:

```text
--tsinfer-path "{params.tsinfer_path}"
```

Do not change the version-labelled filenames.

## 6. Remove legacy virtualenv activation

Delete:

```bash
source {config[env_dir]}/bin/activate
```

from the `generate_ancestors` rule.

Do not replace `env_dir` with another environment-path setting.

The intended way to launch the workflow is:

```bash
uv run snakemake ...
```

Snakemake's shell subprocesses then inherit the project environment.

Inside the rule, continue using:

```bash
python scripts/generate_ancestors.py ...
```

Do not add nested `uv run` commands to individual rules.

## 7. Simplify `scripts/generate_ancestors.py`

Replace the CLI option:

```text
--version
```

with:

```text
--tsinfer-path
```

Remove the entire hard-coded conditional that currently chooses between tsinfer directories according to names such as the old release labels.

Remove the now-unused `os` import.

The tsinfer-selection code should be equivalent to:

```python
tsinfer_path = Path(tsinfer_path).expanduser()
sys.path.insert(0, str(tsinfer_path))

import tsinfer
```

The dynamic import must remain after the source path is inserted.

Do not otherwise refactor this file.

Specifically, leave unchanged:

- positional input/output/log arguments;
- `--threads`;
- `--data-dir`;
- `VariantData` construction;
- ancestral-state handling;
- genotype encoding;
- progress logging;
- `generate_ancestors`;
- output validation.

## 8. Single-version support

Do not create a separate mode or boolean.

This should work naturally:

```yaml
tsinfer:
  - version: "0.5.1"
    path: "~/bin/tsinfer_0_5_1"
```

In that case the workflow should simply:

1. generate one ancestor dataset;
2. load one ancestor dataset;
3. build a one-entry version-to-`AncestorData` map;
4. process it through the existing comparison/dataframe pipeline;
5. produce output rows for that single version.

Do not modify `lib/utils.py` to support this unless a test demonstrates an actual problem. Its existing collection-based behaviour should already accommodate one item.

## 9. Verify removed mechanisms

Run:

```bash
rg -n 'config\["versions"\]|tsinfer_dir|env_dir|tsinfer-old' \
    Snakefile config.yaml.example scripts
```

There should be no remaining executable/configuration references.

Then:

```bash
rg -n -- '--version|--tsinfer-path' Snakefile scripts
```

The ancestor-generation selection mechanism should use `--tsinfer-path`.

Do not mistake unrelated uses of the word "version" for a problem: versions still legitimately appear in filenames, wildcards, dataframe records and the new configuration.

## 10. Verify both configured source installations again

Run the independent import/version checks for:

```text
~/bin/tsinfer_0_5_1 -> 0.5.1
~/bin/tsinfer_0_4_1 -> 0.4.1
```

Do not rebuild them unnecessarily.

## 11. Project validation

Run:

```bash
uv run pytest
uv run ruff check .
```

Do not run the full Snakemake workflow.

Do not run a Snakemake dry run in this phase.

Do not make unrelated fixes.

## 12. Stop before committing

Do **not** run `git commit`.

At completion, report:

1. files changed;
2. the final `tsinfer` section of `config.yaml.example`;
3. whether an ignored local `config.yaml` was migrated;
4. every Snakefile location changed from the old `versions` mechanism;
5. confirmation that ancestor/log filename conventions remain unchanged;
6. the new `generate_ancestors.py` CLI;
7. confirmation that both source installations still import as the expected releases;
8. pytest result;
9. ruff result;
10. `git status --short`;
11. confirmation that no unrelated changes were made.

Then stop.

I will inspect and commit Part 2 myself.

---

# Non-goals for both parts

Do not:

- alter `lib/errors.py`;
- alter `lib/simulation.py`;
- alter `lib/utils.py`;
- change ancestor comparison metrics;
- change output naming conventions except that the exact release strings are now `0.5.1` and `0.4.1`;
- modify simulation expansion;
- modify error-profile expansion;
- change resource configuration;
- install either tsinfer release into the pipeline environment as a package;
- add `tsinfer` itself to `pyproject.toml`;
- create separate Python virtual environments for the releases;
- introduce Snakemake conda environments;
- introduce containers;
- introduce a config abstraction;
- add backwards compatibility for the old config format;
- refactor unrelated Snakefile code;
- make git commits.
