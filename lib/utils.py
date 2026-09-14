import numpy as np
import sgkit
import xarray as xr
import pandas as pd
import json
import tskit
import csv
import math
import warnings
from tqdm import tqdm


def prune_arg(arg):
    mutations_count = np.bincount(arg.mutations_site, minlength=arg.num_sites)
    recurrent = np.where(mutations_count > 1)[0]
    muts_to_remove = []
    for site_id in recurrent:
        muts = arg.site(site_id).mutations
        assert len(muts) > 1
        for mut in muts[1:]:
            muts_to_remove.append(mut.id)
    tables = arg.dump_tables()
    mutations = tables.mutations
    tables.mutations.keep_rows(
        np.isin(range(len(mutations)), muts_to_remove, invert=True)
    )
    return tables.tree_sequence()

def add_zarr_variables(ds, output_path):
    G = ds.call_genotype
    ac = np.sum(G, axis=(1, 2))
    an = G.shape[1]*2 #num_samples*2
    af = ac / an
    assert np.all(af <= 1)
    variables = {
        "variant_allele_count": ac,
        "variant_allele_frequency": af,
        "variant_singleton_mask": ac == 1,
        "variant_ancestral_state": ds.variant_allele[:, 0],
    }
    arrays = {
            name: xr.DataArray(data, dims=["variants"], name=name)
            for name, data in variables.items()
        }
    ds.update(arrays)

    sgkit.save_dataset(
            ds.drop_vars(set(ds.data_vars) - set(arrays.keys())),
            output_path.parent,
            mode="a",
            consolidated=False,
        )
    output_path.touch()

def get_carrier_mrca(site, ts, v):
    """
    Return the mutation node for non-recurrent sites and, for recurrent sites,
    the MRCA of the samples carrying the non-ancestral allele.
    """
    if len(site.mutations) == 0:
        raise ValueError(f"Site {site.id} has no mutations")
    if len(site.mutations) == 1:
        return site.mutations[0].node
        
    tree = ts.at(site.position)
    ancestral_state = site.ancestral_state
    v.decode(site.id)
    derived = next(a for a in v.alleles if a is not None and a != ancestral_state)
    carriers = v.samples[v.genotypes == v.alleles.index(derived)]
    if len(carriers) == 0:
        warnings.warn(
            f"No carriers of the mutation at site {site.id} exist in the true ARG",
            stacklevel=2,
        )
        return None
    else:
        return tree.mrca(*sorted(carriers))
    
def build_ancestor_chunks(anc_data_list, ts, output_dir, chunk_size, metadata_path):
    base_anc_data = anc_data_list[0]
    for anc_data in anc_data_list[1:]:
        assert base_anc_data.num_ancestors == anc_data.num_ancestors
        assert base_anc_data.sequence_length == anc_data.sequence_length
        assert np.array_equal(base_anc_data.sites_position, anc_data.sites_position)
        for sites_1, sites_2 in zip(
            base_anc_data.ancestors_focal_sites, anc_data.ancestors_focal_sites
        ):
            assert np.array_equal(sites_1, sites_2)
            
    records = []
    inf_sites_pos = np.append(base_anc_data.sites_position, base_anc_data.sequence_length)
    ts_sites_pos = np.append(ts.sites_position, ts.sequence_length)
    variant = tskit.Variant(ts, isolated_as_missing=False)
    for inf_node, sites in enumerate(base_anc_data.ancestors_focal_sites):
        for inf_site_id in sites:
            pos = inf_sites_pos[inf_site_id]
            true_site_id = np.searchsorted(ts_sites_pos, pos)
            site = ts.site(true_site_id)
            true_node = get_carrier_mrca(site, ts, variant)
            if true_node is None:
                continue
            records.append(
                {
                    "inf_focal_site": inf_site_id,
                    "true_focal_site": true_site_id,
                    "focal_position": pos,
                    "inf_node": inf_node,
                    "true_node": true_node,
                }
            )

    df = pd.DataFrame.from_records(records)
    assert len(df) > 0
    assert not df.isnull().values.any()

    index = df.index.to_numpy()
    index_chunks = [index[i : i + chunk_size] for i in range(0, len(index), chunk_size)]

    for chunk_id, chunk in enumerate(index_chunks):
        chunk_df = df.loc[chunk]
        assert chunk_size >= len(chunk_df) > 0
        chunk_path = output_dir / f"unprocessed-chunk-{chunk_id}.csv"
        chunk_df.to_csv(chunk_path, index=False)

    with open(metadata_path, "w") as meta_file:
        json.dump({"num_chunks": len(index_chunks)}, meta_file)

def build_shared_site_maps(ts, anc_data_map):
    """
    Build arrays of indices into each sites_position array that correspond to shared sites
    across all ancestor data and the true tree sequence.
    """
    shared_pos = ts.sites_position
    last_pos = ts.sites_position[-1]+1

    for version, anc_data in anc_data_map.items():
        shared_pos = np.intersect1d(shared_pos, anc_data.sites_position, assume_unique=True)

    true_shared_idx = np.searchsorted(ts.sites_position, shared_pos)
    anc_shared_idx_maps = {
        v: np.searchsorted(anc.sites_position, shared_pos) for v, anc in anc_data_map.items()
    }
    shared_pos = np.append(shared_pos, last_pos)
    return shared_pos, true_shared_idx, anc_shared_idx_maps

def process_ancestor_chunk(
    df,
    ts,
    ds,
    anc_data_map,
    rep,
    error_profile,
    empgeno,
    phase,
    mispol,
    output_path,
):
    shared_pos, true_shared_idx, anc_shared_idx_map = build_shared_site_maps(
        ts, anc_data_map
    )
    true_nodes = np.unique(df.true_node)
    print(f"[INFO] Building expanded TS", flush=True)
    tables = ts.dump_tables()
    flags = tables.nodes.flags
    flags[:] = tskit.NODE_IS_SAMPLE
    tables.nodes.flags = flags
    expanded_ts = tables.tree_sequence()
    print(f"[INFO] Generating genotype matrix", flush=True)
    true_genotypes = expanded_ts.genotype_matrix(samples=true_nodes).T
    assert true_genotypes.shape[0] == len(true_nodes)
    true_index_map = {true_node: i for i, true_node in enumerate(true_nodes)}
    
    ds_variant_pos = ds.variant_position.values
    include_mispol = ~ds.variant_mispolarisation_mask.values
    ds_shared_idx = np.searchsorted(ds_variant_pos, shared_pos[:-1])
    assert np.array_equal(ds_variant_pos[ds_shared_idx], shared_pos[:-1])
    shared_include_mispol = include_mispol[ds_shared_idx].astype("int8")
    allele_frequency = ds.variant_allele_frequency.values
    geno_error_count = ds.variant_genotype_error_count.values
    print(f"[INFO] Building dataframe", flush=True)

    with open(output_path, "w", newline="") as f:
        writer = None
        for row in tqdm(
            df.itertuples(index=False),
            total=len(df),
            desc="Processing rows",
            ncols=80,
            mininterval=5,
        ):
            true_node = row.true_node
            true_node_index = true_index_map[true_node]
            a = true_genotypes[true_node_index]
            a_shared = a[true_shared_idx]
            segment = np.where(a_shared != tskit.MISSING_DATA)[0]
            assert len(segment) > 0
            true_left = segment[0]
            true_right = segment[-1] + 1
            true_full_haplotype = np.bitwise_xor(
                (a_shared > 0).astype("int8"), shared_include_mispol
            )
            true_time = ts.nodes_time[true_node]
            inf_node = row.inf_node
            true_boundary = {}
            true_boundary["left"] = shared_pos[true_left]
            true_boundary["right"] = shared_pos[true_right]
            true_span = true_boundary["right"] - true_boundary["left"]

            olap_left = true_left
            olap_right = true_right
            anc_dict = {}
            anc_interval_dict = {}
            for version, anc_data in anc_data_map.items():
                anc = anc_data.ancestor(inf_node)
                anc_dict[version] = anc
                anc_shared_idx = anc_shared_idx_map[version]
                anc_left  = np.count_nonzero(anc_shared_idx < anc.start)
                anc_right = np.count_nonzero(anc_shared_idx < anc.end)
                assert anc_left < anc_right
                anc_interval_dict[version] = (anc_left, anc_right)
                olap_left = max(olap_left, anc_left)
                olap_right = min(olap_right, anc_right)
            assert olap_left < olap_right
            olap_boundary = {}
            olap_boundary["left"] = shared_pos[olap_left]
            olap_boundary["right"] = shared_pos[olap_right]
            olap_site_span = olap_right - olap_left
            olap_span = olap_boundary["right"] - olap_boundary["left"]
            true_olap = true_full_haplotype[olap_left:olap_right]
            for version, anc in anc_dict.items():
                anc_shared_idx = anc_shared_idx_map[version]
                inf_left, inf_right = anc_interval_dict[version]
                inf_haplotype = anc.full_haplotype[anc_shared_idx]
                inf_olap = inf_haplotype[olap_left:olap_right]
                assert len(inf_olap) == len(true_olap)
                errors = inf_olap != true_olap
                should_be_0 = true_olap & ~inf_olap
                should_be_1 = ~true_olap & inf_olap
                inferred_boundary = {}
                inferred_boundary["left"] = shared_pos[inf_left]
                inferred_boundary["right"]  = shared_pos[inf_right]
                inferred_span = inferred_boundary["right"] - inferred_boundary["left"]
                num_errors = np.sum(errors)
                num_should_be_0 = np.sum(should_be_0)
                num_should_be_1 = np.sum(should_be_1)
                ds_focal_site = np.searchsorted(ds_variant_pos, row.focal_position)
                af = allele_frequency[ds_focal_site]
                #assert math.isclose(af, anc.time, rel_tol=1e-4)
                for side in ["left", "right"]:
                    record = {
                        "inferred_node": inf_node,
                        "true_node": true_node,
                        "replicate": rep,
                        "error_profile": error_profile,
                        "empgeno": empgeno,
                        "phase": phase,
                        "mispol": mispol,
                        "version": version,
                        "side": side,
                        "inf_focal_site": row.inf_focal_site,
                        "true_focal_site": row.true_focal_site,
                        "focal_position": row.focal_position,
                        "focal_site_mispolarised": include_mispol[ds_focal_site],
                        "focal_site_genotype_error_count": geno_error_count[ds_focal_site],
                        "allele_frequency": af,
                        "true_time": true_time,
                        "inferred_time": anc.time,
                        "true_boundary": true_boundary[side],
                        "true_span": true_span,
                        "inferred_boundary": inferred_boundary[side],
                        "inferred_span": inferred_span,
                        "overlap_boundary": olap_boundary[side],
                        "overlap_span": olap_span,
                        "overlap_site_span": olap_site_span,
                        "num_errors": num_errors,
                        "num_errors_per_bp": num_errors/olap_span,
                        "num_errors_per_site": num_errors/olap_site_span,
                        "num_should_be_0": num_should_be_0,
                        "num_should_be_1": num_should_be_1,
                    }
                    if side == "left":
                        overshoot = true_boundary["left"] - inferred_boundary["left"]
                    else:
                        overshoot = inferred_boundary["right"] - true_boundary["right"]
                    record.update({"overshoot": overshoot})
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=record.keys())
                        writer.writeheader()
                    writer.writerow(record)

    print(f"[INFO] Finished writing chunk {output_path}")
