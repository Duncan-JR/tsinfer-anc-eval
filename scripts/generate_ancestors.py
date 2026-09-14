import sys
import warnings
from pathlib import Path

import click

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*LMDBStore is deprecated.*",
)


@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.argument("log", type=click.Path())
@click.option("--tsinfer-path", required=True, type=click.Path(exists=True))
@click.option("--threads", required=True, type=int)
@click.option("--data-dir", required=True, type=click.Path())
def generate_ancestors(input, output, log, tsinfer_path, threads, data_dir):
    """Generate ancestors from a tree sequence."""

    sys.path.insert(0, tsinfer_path)
    import tsinfer

    data_dir = Path(data_dir)
    vdata = tsinfer.VariantData(
        input.replace(".mods_done", ""),
        ancestral_state="variant_mispolarised_ancestral_state",
    )
    assert vdata.num_sites > 0

    with open(log, "w") as log_f:
        ancestors = tsinfer.generate_ancestors(
            vdata,
            path=output,
            genotype_encoding=1,
            num_threads=threads,
            progress_monitor=tsinfer.progress.ProgressMonitor(
                tqdm_kwargs={"file": log_f, "mininterval": 30}
            ),
        )
        if ancestors.num_ancestors == 0:
            raise ValueError("No ancestors generated")
        if ancestors.num_sites == 0:
            raise ValueError("No sites generated")


if __name__ == "__main__":
    generate_ancestors()
