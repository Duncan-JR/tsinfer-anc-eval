import yaml
import sys
import json
import tskit
import pandas as pd
import numpy as np
import shutil
import warnings
import math
import contextlib
import xarray as xr
import sgkit
import stdpopsim
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, message=".*LMDBStore*")
configfile: "config.yaml"

sys.path.append(config["tsinfer_dir"])
import tsinfer
from lib import simulation, utils, errors

shell.prefix(config["prefix"])
data_dir = Path(config["data_dir"])
progress_dir = Path(config["progress_dir"])

def get_resource(rule_name, resource_type):
    if (
        rule_name in config["resources"]
        and resource_type in config["resources"][rule_name]
    ):
        return config["resources"][rule_name][resource_type]
    else:
        return config["resources"]["default"][resource_type]


def sim_metadata():
    for sim in config["sims"]:
        model = sim["model"]
        contig = sim["contig"]
        left = sim["left"]
        right = sim["right"]
        samples = sim["samples"]
        n = sum(samples.values())
        seed = sim["seed"]
        for rep in range(sim["num_reps"]):
            yield model, contig, left, right, n, seed, rep


def expand_data_frames():
    return [
        data_dir
        / "dataframes"
        / f"{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-aggregated.csv"
        for model, contig, left, right, n, seed, rep in sim_metadata()
    ]

def map_error_profiles(empgeno, phase, mispol):
    output = "unknown"
    for err_conf in config["error_configs"]:
        _empgeno = err_conf["empgeno"]
        _phase = float(err_conf["phase"])
        _mispol = float(err_conf["mispol"])
        _name = err_conf["name"]
        if (
            _empgeno == empgeno
            and math.isclose(_phase, float(phase), rel_tol=1e-4)
            and math.isclose(_mispol, float(mispol), rel_tol=1e-4)
        ):
            output = _name
    return output
        

rule all:
    input:
        expand_data_frames(),


rule simulate:
    output:
        data_dir / "simulated" / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}.trees",
    threads: get_resource("simulate", "threads")
    resources:
        mem_mb=get_resource("simulate", "mem_mb"),
        time_min=get_resource("simulate", "time_min"),
    run:
        def match_sim_entry(wildcards):
            for sim in config["sims"]:
                n = sum(sim["samples"].values())   
                if (
                    sim["model"] == wildcards.model
                    and sim["contig"] == wildcards.contig
                    and math.isclose(float(sim["left"]), float(wildcards.left), rel_tol=1e-4)
                    and math.isclose(float(sim["right"]), float(wildcards.right), rel_tol=1e-4)
                    and n == int(wildcards.n)
                    and int(sim["seed"]) == int(wildcards.seed)
                ):
                    for rep in range(sim["num_reps"]):
                        return sim
            raise ValueError(f"No matching sim entry for wildcards: {wildcards}")

        sim_entry = match_sim_entry(wildcards)
        samples = sim_entry["samples"]
        arg = simulation.simulate(
            model=wildcards.model,
            contig=wildcards.contig,
            samples=samples,
            left=float(wildcards.left),
            right=float(wildcards.right),
            seed=int(wildcards.seed) + int(wildcards.rep),
        )
        arg.dump(output[0])


# rule prune_simulated_ts:
#     input:
#         data_dir / "simulated" / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-raw.trees",
#     output:
#         data_dir
#         / "simulated"
#         / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-pruned.trees",
#     threads: get_resource("prune_simulated_ts", "threads")
#     resources:
#         mem_mb=get_resource("prune_simulated_ts", "mem_mb"),
#         time_min=get_resource("prune_simulated_ts", "time_min"),
#     run:
#         arg = tskit.load(input[0])
#         arg = utils.prune_arg(arg)
#         arg.dump(output[0])

rule bio2zarr_convert:
    input:
        data_dir
        / "simulated"
        / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}.trees",
    output:
        data_dir
        / "zarr_vcfs"
        / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-raw.zarr"
        / ".vcf_done",
    threads: get_resource("bio2zarr_convert", "threads")
    resources:
        mem_mb=get_resource("bio2zarr_convert", "mem_mb"),
        time_min=get_resource("bio2zarr_convert", "time_min"),
    run:
        from bio2zarr import tskit as ts2zarr
        import shutil

        arg_path = input[0]
        zarr_path = Path(output[0]).parent
        # Snakemake makes the directory first so we have to remove it
        if zarr_path.exists():
            shutil.rmtree(zarr_path)

        ts2zarr.convert(
            arg_path,
            zarr_path,
            worker_processes=threads,
        )
        Path(output[0]).touch()


def ds_dir(wildcards):
    model = wildcards.model
    contig = wildcards.contig
    left = wildcards.left
    right = wildcards.right
    n = wildcards.n
    seed = wildcards.seed
    rep = wildcards.rep
    
    return data_dir / "zarr_vcfs" / f"{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-raw.zarr"


rule add_zarr_variables:
    input:
        lambda wildcards: ds_dir(wildcards) / ".vcf_done",
    output:
        data_dir
        / "zarr_vcfs"
        / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-raw.zarr"
        / ".mods_done",
    threads: get_resource("add_zarr_variables", "threads")
    resources:
        mem_mb=get_resource("add_zarr_variables", "mem_mb"),
        time_min=get_resource("add_zarr_variables", "time_min"),
        runtime=get_resource("add_zarr_variables", "time_min"),
    run:
        ds = sgkit.load_dataset(Path(input[0]).parent, consolidated=False)
        utils.add_zarr_variables(
            ds=ds,
            output_path=Path(output[0]),
        )

rule add_errors:
    input:
        data_dir
        / "zarr_vcfs"
        / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-raw.zarr"
        / ".mods_done",
    output:
        data_dir
        / "zarr_vcfs"
        / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-empgeno-{empgeno}-phase{phase}-mispol{mispol}.zarr"
        / ".mods_done",
    threads: get_resource("add_genotype_errors", "threads")
    resources:
        mem_mb=get_resource("add_genotype_errors", "mem_mb"),
        time_min=get_resource("add_genotype_errors", "time_min"),
    run:
        output_path = Path(output[0])
        ds = sgkit.load_dataset(Path(input[0]).parent, consolidated=False)
        error_csv_path = config["error_probs_path"]
        empirical_errors_enabled = wildcards.empgeno
        phase_error_rate = float(wildcards.phase)
        mispol_error_rate = float(wildcards.mispol)
        assert empirical_errors_enabled in {"on", "off"}
        assert 0 <= phase_error_rate <= 1
        assert 0 <= mispol_error_rate <= 1
        seed = int(wildcards.seed) + int(wildcards.rep)
        errors.add_errors(
            ds=ds,
            output_path=output_path,
            error_csv_path=error_csv_path,
            empirical_errors_enabled=empirical_errors_enabled,
            phase_error_rate=phase_error_rate,
            mispol_error_rate=mispol_error_rate,
            seed=seed,
        )

def zarr_with_errors(wildcards):
    model = wildcards.model
    contig = wildcards.contig
    left = wildcards.left
    right = wildcards.right
    n = wildcards.n
    seed = wildcards.seed
    rep = wildcards.rep
    empgeno = wildcards.empgeno
    phase = wildcards.phase
    mispol = wildcards.mispol
    return data_dir / "zarr_vcfs" / f"{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-empgeno-{empgeno}-phase{phase}-mispol{mispol}.zarr"

rule generate_ancestors:
    input:
        lambda wildcards: zarr_with_errors(wildcards) / ".mods_done",
    output:
        data_dir
        / "ancestors"
        / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-empgeno-{empgeno}-phase{phase}-mispol{mispol}-v{version}-ancestors.zarr",
    log:
        progress_dir
        / "generate_ancestors"
        / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-empgeno-{empgeno}-phase{phase}-mispol{mispol}-v{version}.log",
    threads: get_resource("generate_ancestors", "threads")
    resources:
        mem_mb=get_resource("generate_ancestors", "mem_mb"),
        time_min=get_resource("generate_ancestors", "time_min"),
    shell:
        """
        source {config[env_dir]}/bin/activate
        python scripts/generate_ancestors.py \
            {input} \
            {output} \
            {log} \
            --version {wildcards.version} \
            --threads {threads} \
            --data-dir {config[data_dir]}
        """

def expand_ancestors_by_version(wildcards):
    model = wildcards.model
    contig = wildcards.contig
    left = wildcards.left
    right = wildcards.right
    n = wildcards.n
    seed = wildcards.seed
    rep = wildcards.rep
    empgeno = wildcards.empgeno
    phase = wildcards.phase
    mispol = wildcards.mispol

    return [
        data_dir / "ancestors" / f"{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-empgeno-{empgeno}-phase{phase}-mispol{mispol}-v{version}-ancestors.zarr"
        for version in config["versions"]
    ]

checkpoint build_ancestor_chunks:
    input:
        data_dir
        / "simulated"
        / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}.trees",
        expand_ancestors_by_version,
    output:
        data_dir
        / "chunks"
        / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-empgeno-{empgeno}-phase{phase}-mispol{mispol}"
        / "metadata.json",
    threads: get_resource("build_ancestor_chunks", "threads")
    resources:
        mem_mb=get_resource("build_ancestor_chunks", "mem_mb"),
        time_min=get_resource("build_ancestor_chunks", "time_min"),
    run:
        anc_data_list = [tsinfer.formats.AncestorData.load(path) for path in input[1:]]
        assert len(anc_data_list) == len(config["versions"])
        ts = tskit.load(input[0])
        metadata_path = Path(output[0])
        output_dir = metadata_path.parent
        chunk_size = config["ancestor_chunk_size"]
        utils.build_ancestor_chunks(
            anc_data_list=anc_data_list,
            ts=ts,
            metadata_path=metadata_path,
            output_dir=output_dir,
            chunk_size=chunk_size,
        )

def chunk_input(wildcards):
    model = wildcards.model
    contig = wildcards.contig
    left = wildcards.left
    right = wildcards.right
    n = wildcards.n
    seed = wildcards.seed
    rep = wildcards.rep
    empgeno = wildcards.empgeno
    phase = wildcards.phase
    mispol = wildcards.mispol
    chunk_id = wildcards.chunk_id
    return (
        data_dir 
        / "chunks" 
        / f"{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-empgeno-{empgeno}-phase{phase}-mispol{mispol}" 
        / f"unprocessed-chunk-{chunk_id}.csv"
    )


rule process_ancestor_chunk:
    input:
        chunk=chunk_input,
        arg_path=data_dir
        / "simulated"
        / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}.trees",
        anc_data_paths=expand_ancestors_by_version,
        zarr_path=lambda wildcards: zarr_with_errors(wildcards) / ".mods_done",
    output:
        data_dir 
        / "chunks" 
        / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-empgeno-{empgeno}-phase{phase}-mispol{mispol}" 
        / "processed-chunk-{chunk_id}.csv"
    log:
        progress_dir 
        / "process_chunks" 
        / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-empgeno-{empgeno}-phase{phase}-mispol{mispol}" 
        / "processed-chunk-{chunk_id}.log"
    threads: get_resource("process_ancestor_chunk", "threads")
    resources:
        mem_mb=get_resource("process_ancestor_chunk", "mem_mb"),
    run:
        with open(log[0], "w") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(
                log_file
            ):
                print(f"[INFO] Loading chunk {wildcards.chunk_id}", flush=True)
                chunk_path = Path(input.chunk)
                df = pd.read_csv(chunk_path)
                assert len(df) > 0
                print(f"[INFO] Loading tree sequence", flush=True)
                ts = tskit.load(input.arg_path)
                versions = config["versions"]
                print(f"[INFO] Importing AncestorData", flush=True)
                anc_data_map = {
                    version: tsinfer.formats.AncestorData.load(path)
                    for path, version in zip(input.anc_data_paths, versions)
                }
                print(f"[INFO] Starting DF generation", flush=True)
                ds = sgkit.load_dataset(Path(input.zarr_path).parent, consolidated=False)
                empgeno = wildcards.empgeno
                phase = float(wildcards.phase)
                mispol = float(wildcards.mispol)
                error_profile = map_error_profiles(
                    empgeno=empgeno,
                    phase=phase,
                    mispol=mispol,
                )
                assert error_profile != "unknown", (
                    "Error profile not found for "
                    f"empgeno={empgeno}, phase={phase}, mispol={mispol}"
                )
                utils.process_ancestor_chunk(
                    df=df,
                    ts=ts,
                    ds=ds,
                    anc_data_map=anc_data_map,
                    rep=wildcards.rep,
                    error_profile=error_profile,
                    empgeno=wildcards.empgeno,
                    phase=wildcards.phase,
                    mispol=wildcards.mispol,
                    output_path=output[0],
                )

def get_checkpoint(wildcards):
    cp = checkpoints.build_ancestor_chunks.get(
        model=wildcards.model,
        contig=wildcards.contig,
        left=wildcards.left,
        right=wildcards.right,
        n=wildcards.n,
        seed=wildcards.seed,
        rep=wildcards.rep,
        empgeno=wildcards.empgeno,
        phase=wildcards.phase,
        mispol=wildcards.mispol,
    )
    return cp

def chunk_ids(wildcards):
    cp = get_checkpoint(wildcards)
    with open(cp.output[0]) as f:
        metadata = json.load(f)
    return list(range(metadata["num_chunks"]))

rule process_all_chunks:
    input:
        lambda wildcards: expand(
            data_dir 
            / "chunks" 
            / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-empgeno-{empgeno}-phase{phase}-mispol{mispol}" 
            / "processed-chunk-{chunk_id}.csv",
            model=wildcards.model,
            contig=wildcards.contig,
            left = wildcards.left,
            right = wildcards.right,
            n = wildcards.n,
            seed = wildcards.seed,
            rep=wildcards.rep,
            empgeno=wildcards.empgeno,
            phase=wildcards.phase,
            mispol=wildcards.mispol,
            chunk_id=chunk_ids(wildcards),
        ),
    output:
        temp(
            data_dir 
            / "chunks" 
            / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-empgeno-{empgeno}-phase{phase}-mispol{mispol}" 
            / ".processed"
        ),
    run:
        Path(output[0]).touch()


def aggregate_chunk_paths(wildcards):
    cp = get_checkpoint(wildcards)
    chunk_dir = Path(cp.output[0]).parent
    return sorted(str(p) for p in chunk_dir.glob("processed-chunk-*.csv"))

 
checkpoint aggregate_ancestor_chunks:
    input:
        processed=data_dir
        / "chunks"
        / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-empgeno-{empgeno}-phase{phase}-mispol{mispol}" 
        / ".processed",
        chunks=aggregate_chunk_paths,
    output:
        data_dir
        / "dataframes"
        / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-empgeno-{empgeno}-phase{phase}-mispol{mispol}-ancestors.csv",
    threads: get_resource("aggregate_ancestor_chunks", "threads")
    resources:
        mem_mb=get_resource("aggregate_ancestor_chunks", "mem_mb"),
        time_min=get_resource("aggregate_ancestor_chunks", "time_min"),
    run:
        dfs = [pd.read_csv(p) for p in input.chunks]
        pd.concat(dfs, ignore_index=True).to_csv(output[0], index=False)


def aggregated_dataframe_paths(wildcards):
    paths = []
    for model, contig, left, right, n, seed, rep in sim_metadata():
        if (
            model == wildcards.model
            and contig == wildcards.contig
            and math.isclose(float(left), float(wildcards.left), rel_tol=1e-4)
            and math.isclose(float(right), float(wildcards.right), rel_tol=1e-4)
            and int(n) == int(wildcards.n)
            and int(seed) == int(wildcards.seed)
        ):
            for err_conf in config["error_configs"]:
                empgeno = err_conf["empgeno"]
                phase = err_conf["phase"]
                mispol = err_conf["mispol"]
                paths.append(
                    data_dir
                    / "dataframes"
                    / f"{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-rep{rep}-empgeno-{empgeno}-phase{phase}-mispol{mispol}-ancestors.csv"
                )
    
    assert len(paths) > 0
    return paths

    
rule finalize_all_dataframes:
    input:
        aggregated_dataframe_paths
    output:
        temp(
            data_dir
            / "dataframes"
            / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}.finalized"
        )
    run:
        Path(output[0]).touch()

rule aggregate_dataframes:
    input:
        finalized=data_dir
        / "dataframes"
        / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}.finalized",
        dfs=aggregated_dataframe_paths,
    output:
        data_dir
        / "dataframes"
        / "{model}-{contig}-L{left}-R{right}-n{n}-s{seed}-aggregated.csv",
    threads: get_resource("aggregate_dataframes", "threads")
    resources:
        mem_mb=get_resource("aggregate_dataframes", "mem_mb"),
        time_min=get_resource("aggregate_dataframes", "time_min"),
    run:
        dfs = [pd.read_csv(p) for p in input.dfs]
        pd.concat(dfs, ignore_index=True).to_csv(output[0], index=False)
