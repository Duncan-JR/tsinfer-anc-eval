
# PART 1 — Install the tagged tsinfer releases and prepare dependencies

## Goal

Prepare two independent source/build installations:

```text
~/bin/tsinfer_0_5_1
~/bin/tsinfer_0_4_1
```

using the exact released tags:

- `0.5.1`
- `0.4.1`

Compile both C extensions using the Python environment managed by the `tsinfer-anc-eval` uv project.

Also add any runtime dependencies needed because these source trees will be imported directly rather than installed as Python packages.

Do **not** modify the Snakefile or `scripts/generate_ancestors.py` in this part.

Do **not** commit anything.

## 1. Start from the pipeline repository

Run from the root of `tsinfer-anc-eval`.

Synchronise the existing project:

```bash
uv sync
```

Record its absolute path:

```bash
PIPELINE_ROOT="$PWD"
```

## 2. Add runtime dependencies required by source-tree imports

The pipeline already supplies many shared dependencies, including NumPy, numba, tqdm, tskit and zarr.

Because the two tsinfer checkouts will be imported directly from their source directories rather than installed into `.venv`, ensure the project's uv environment also contains the remaining runtime dependencies required by the release tags.

Add:

```bash
uv add six humanize daiquiri lmdb sortedcontainers "attrs>=19.2.0" "psutil>=5.9.0"
```

Do **not** add `tsinfer` itself to `pyproject.toml`.

This is expected to modify:

```text
pyproject.toml
uv.lock
```

Then run:

```bash
uv sync
```

## 3. Clone the exact releases

Create the parent directory if needed:

```bash
mkdir -p ~/bin
```

Clone `0.5.1`:

```bash
git clone \
    --branch 0.5.1 \
    --depth 1 \
    https://github.com/tskit-dev/tsinfer.git \
    ~/bin/tsinfer_0_5_1
```

Clone `0.4.1`:

```bash
git clone \
    --branch 0.4.1 \
    --depth 1 \
    https://github.com/tskit-dev/tsinfer.git \
    ~/bin/tsinfer_0_4_1
```

If either target directory already exists, **do not delete or overwrite it automatically**.

Instead inspect it and reuse it only if it is the correct tsinfer repository at the exact requested tag.

Verify:

```bash
git -C ~/bin/tsinfer_0_5_1 describe --tags --exact-match
git -C ~/bin/tsinfer_0_4_1 describe --tags --exact-match
```

Expected:

```text
0.5.1
0.4.1
```

## 4. Compile the C extensions

Both releases require the compiled `_tsinfer` extension.

Compile each checkout in place using the `tsinfer-anc-eval` uv environment rather than using the historical Makefiles.

For `0.5.1`:

```bash
uv run \
    --project "$PIPELINE_ROOT" \
    --directory "$HOME/bin/tsinfer_0_5_1" \
    --with setuptools \
    python setup.py build_ext --inplace
```

For `0.4.1`:

```bash
uv run \
    --project "$PIPELINE_ROOT" \
    --directory "$HOME/bin/tsinfer_0_4_1" \
    --with setuptools \
    python setup.py build_ext --inplace
```

A working C compiler is required.

Do not install either checkout with:

```text
pip install
uv pip install
uv add tsinfer
editable installation
```

The intended arrangement is that each source directory itself contains its matching compiled `_tsinfer` extension.

## 5. Verify both releases independently

Use separate Python processes so Python module caching cannot cause one release to masquerade as the other.

### 0.5.1

```bash
uv run python -c '
import sys
from pathlib import Path

path = Path("~/bin/tsinfer_0_5_1").expanduser()
sys.path.insert(0, str(path))

import tsinfer
import _tsinfer

print(tsinfer.__version__)
print(tsinfer.__file__)
print(_tsinfer.__file__)

assert tsinfer.__version__ == "0.5.1"
'
```

### 0.4.1

```bash
uv run python -c '
import sys
from pathlib import Path

path = Path("~/bin/tsinfer_0_4_1").expanduser()
sys.path.insert(0, str(path))

import tsinfer
import _tsinfer

print(tsinfer.__version__)
print(tsinfer.__file__)
print(_tsinfer.__file__)

assert tsinfer.__version__ == "0.4.1"
'
```

Both imports must resolve to their respective `~/bin` directories.

## 6. Validate the pipeline project

Run:

```bash
uv run pytest
uv run ruff check .
```

Do not make unrelated lint or test fixes.

## 7. Stop before committing

Do **not** run `git commit`.

At the end, report:

1. whether each tag was cloned successfully;
2. the exact checkout paths;
3. whether each C extension compiled successfully;
4. for each release:
   - `tsinfer.__version__`
   - `tsinfer.__file__`
   - `_tsinfer.__file__`
5. changes made to `pyproject.toml`;
6. whether `uv.lock` changed;
7. pytest result;
8. ruff result;
9. `git status --short`.

Then stop.

I will review and commit this part myself.

