import os
import sys
import click
import tskit
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*LMDBStore is deprecated.*")

@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.argument("log", type=click.Path())
@click.option("--version", required=True, type=str)
@click.option("--threads", required=True, type=int)
@click.option("--data-dir", required=True, type=click.Path())


def generate_ancestors(input, output, log, version, threads, data_dir):
    """Generate ancestors from a tree sequence."""

    if version == "1.0" or version == "new":
        tsinfer_path = os.path.abspath("/well/kelleher/users/uuc395/tsinfer")
    elif version == "0.4" or version == "old":
        tsinfer_path = os.path.abspath("/well/kelleher/users/uuc395/tsinfer-old")
    else:
        raise ValueError("Version must be either 1.0/new or 0.4/old")

    sys.path.append(tsinfer_path)
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


