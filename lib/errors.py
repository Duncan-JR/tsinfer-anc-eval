import numpy as np
import pandas as pd
import zarr
from numba import njit


def scale_genotype_error_probs(probs, geno_multiplier):
    """Scale the cumulative genotype-error hazard independently by row.

    The correct-call probability is raised to ``geno_multiplier`` and the
    empirical proportions of the two error outcomes are preserved.
    """
    if not np.isfinite(geno_multiplier) or geno_multiplier < 0:
        message = "geno_multiplier must be a finite value greater than or equal to 0"
        raise ValueError(message)

    scaled_probs = np.zeros_like(probs, dtype=float)
    for true_genotype in range(3):
        error_genotypes = np.arange(3) != true_genotype
        correct_prob = probs[true_genotype, true_genotype]
        error_prob = 1 - correct_prob
        scaled_error_prob = 1 - correct_prob**geno_multiplier
        if error_prob > 0:
            error_scale = scaled_error_prob / error_prob
            scaled_errors = probs[true_genotype, error_genotypes] * error_scale
            scaled_probs[true_genotype, error_genotypes] = scaled_errors
        scaled_probs[true_genotype, true_genotype] = 1 - scaled_error_prob

    return scaled_probs


def fetch_empirical_probs(freq, df, geno_multiplier=1.0):
    """
    Fetch empirical genotype error probabilities for a given allele frequency.
    The input frequency should be between 0 and 1, inclusive.
    """
    error_freq = df.freq.values
    assert 0 <= freq <= 1
    if freq < error_freq.min():
        freq = error_freq.min()
    elif freq > error_freq.max():
        freq = error_freq.max()
    # Last row has frequency 1.0 exactly, so we can use 'right' to fetch that row
    # correctly.
    row = df.loc[np.searchsorted(error_freq, freq, side="right")]
    genotype_probs = np.array(
        [
            [row.p00, row.p01, row.p02],
            [row.p10, row.p11, row.p12],
            [row.p20, row.p21, row.p22],
        ]
    )
    scaled_probs = scale_genotype_error_probs(genotype_probs, geno_multiplier)
    probs = np.array(
        [
            [
                scaled_probs[0, 0],
                0.5 * scaled_probs[0, 1],
                0.5 * scaled_probs[0, 1],
                scaled_probs[0, 2],
            ],
            [scaled_probs[1, 0], scaled_probs[1, 1], 0, scaled_probs[1, 2]],
            [scaled_probs[1, 0], 0, scaled_probs[1, 1], scaled_probs[1, 2]],
            [
                scaled_probs[2, 0],
                0.5 * scaled_probs[2, 1],
                0.5 * scaled_probs[2, 1],
                scaled_probs[2, 2],
            ],
        ]
    )
    assert np.all(np.isclose(probs.sum(axis=1), 1.0, atol=1e-5))
    return probs


def encode_genotypes(genotypes):
    return genotypes[:, 0] * 2 + genotypes[:, 1]


def decode_genotypes(idx):
    genotypes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
    return genotypes[idx]


def sample_genotype(genotype, probs, rng):
    assert len(genotype) == 2
    input_idx = encode_genotypes(np.array([genotype]))[0]
    output_idx = rng.choice(len(probs), p=probs[input_idx])
    return decode_genotypes(output_idx)


def sample_genotypes_vectorised(genotypes, probs, rng):
    input_idx = encode_genotypes(genotypes)
    cum_probs = np.cumsum(probs, axis=1)
    U = rng.random(len(input_idx))
    output_idx = (U[:, None] < cum_probs[input_idx]).argmax(axis=1)
    return decode_genotypes(output_idx)


def add_empirical_genotype_errors(G_in, rng, probs_func, **kwargs):
    # no multiallelic sites
    assert len(np.unique(G_in)) <= 2
    assert len(G_in.shape) == 3
    assert G_in.shape[2] == 2
    assert G_in.dtype == np.int8

    G_out = np.full_like(G_in, 0, dtype=np.int8)
    num_sites = G_out.shape[0]
    num_samples = G_out.shape[1]
    an = num_samples * 2

    for site in range(num_sites):
        g = G_in[site, :, :]
        freq = np.sum(g) / an
        probs = probs_func(freq, **kwargs)
        G_out[site, :, :] = sample_genotypes_vectorised(g, probs, rng)
    return G_out


@njit
def phase_switch_diplotype(d_in, d_out, phase_array, switch_sites):
    phase = 0
    k = 0
    num_switches = len(switch_sites)
    num_sites = d_out.shape[0]

    for i in range(num_sites):
        if k < num_switches and i == switch_sites[k]:
            phase ^= 1
            k += 1
        phase_array[i] = phase
        if phase == 0:
            d_out[i] = d_in[i]
        else:
            d_out[i] = d_in[i, ::-1]
    return d_out, phase_array


def sample_phase_switches(d_in, phase_ser, rng):
    include_het = d_in.sum(axis=1) == 1
    het_sites = include_het.nonzero()[0]
    switch_candidates = het_sites[1:]

    include_switch_sites = rng.random(len(switch_candidates)) < phase_ser
    num_switches = np.sum(include_switch_sites)
    phase_array = np.zeros(d_in.shape[0], dtype=bool)
    if num_switches > 0:
        # Select the later heterozygote in each consecutive pair.
        switch_sites = switch_candidates[include_switch_sites]
        d_out = np.zeros_like(d_in)
        d_out, phase_array = phase_switch_diplotype(
            d_in, d_out, phase_array, switch_sites
        )
        assert d_out.sum() == d_in.sum()
    else:
        d_out = d_in
    return d_out, phase_array, num_switches


def add_phase_switch_errors(G_in, phase_ser, rng):
    """
    Add random phase switches to every diplotype in the call genotypes array.
    The switch error rate (SER) is the independent switch probability per
    pair of consecutive heterozygotes, allowing intervening homozygous sites.
    """
    assert len(np.unique(G_in)) <= 2
    assert len(G_in.shape) == 3
    assert G_in.shape[2] == 2
    assert G_in.dtype == np.int8

    num_sites = G_in.shape[0]
    num_samples = G_in.shape[1]
    sample_switch_count = np.zeros(num_samples)
    call_genotype_phase = np.zeros([num_sites, num_samples], dtype=bool)
    if phase_ser == 0:
        return G_in, call_genotype_phase, sample_switch_count

    G_out = np.full_like(G_in, 0, dtype=np.int8)
    for sample in range(num_samples):
        d_in = G_in[:, sample, :]
        d_out, phase_array, num_switches = sample_phase_switches(d_in, phase_ser, rng)
        G_out[:, sample, :] = d_out
        call_genotype_phase[:, sample] = phase_array
        sample_switch_count[sample] = num_switches

    return G_out, call_genotype_phase, sample_switch_count


def unbiased_mispolarise(
    variant_allele, ancestral_state, mispol_rate, rng, singleton_mask
):
    assert 0 <= mispol_rate <= 1
    num_sites = len(ancestral_state)
    ancestral, derived = variant_allele[:, 0], variant_allele[:, 1]
    assert np.array_equal(ancestral, ancestral_state)
    singleton_mask = np.asarray(singleton_mask, dtype=bool)
    assert singleton_mask.shape == (num_sites,)
    include_mispol = rng.random(num_sites) < mispol_rate
    include_mispol &= ~singleton_mask
    mispol_ancestral = ancestral_state.copy()
    mispol_ancestral[include_mispol] = derived[include_mispol]

    return include_mispol, mispol_ancestral


def _create_error_array(root, name, dims, dtype, filters=None):
    """Create an xarray-compatible array aligned with the genotype chunks."""
    genotypes = root["call_genotype"]
    genotype_dims = genotypes.attrs["_ARRAY_DIMENSIONS"]
    axes = [genotype_dims.index(dim) for dim in dims]
    shape = tuple(genotypes.shape[axis] for axis in axes)
    chunks = tuple(genotypes.chunks[axis] for axis in axes)
    array = root.create_dataset(
        name,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=genotypes.compressor,
        filters=filters,
        fill_value=None,
    )
    array.attrs["_ARRAY_DIMENSIONS"] = dims
    return array


def add_errors(
    input_path,
    output_path,
    error_csv_path,
    geno_multiplier,
    space_per_switch,
    mispol_rate,
    seed,
    sequence_length,
):
    """Add errors using variant blocks, then sample blocks for phase switches.

    The separate passes preserve the random draw order of the in-memory helpers:
    genotype errors by site, phase switches by sample, then mispolarisation.
    Calibrate SER to the requested mean spacing in bp using the pooled number
    of consecutive heterozygote pairs after genotype errors. Zero disables phase
    switches; sequence_length is the simulated interval length in bp.
    Block widths come from the stored genotype chunks. Peak genotype memory is
    proportional to variants_per_chunk * samples, or (when phasing is enabled)
    variants * samples_per_chunk, rather than variants * samples.
    """
    if not np.isfinite(geno_multiplier) or geno_multiplier < 0:
        message = "geno_multiplier must be a finite value greater than or equal to 0"
        raise ValueError(message)
    if not np.isfinite(space_per_switch) or space_per_switch < 0:
        raise ValueError("space_per_switch must be finite and nonnegative")
    if not np.isfinite(sequence_length) or sequence_length <= 0:
        raise ValueError("sequence_length must be finite and positive")
    if not 0 <= mispol_rate <= 1:
        raise ValueError("mispol_rate must be between 0 and 1")

    source = zarr.open_group(str(input_path), mode="r")
    root = zarr.open_group(str(output_path.parent), mode="w")
    # Copy compressed chunks directly, without decoding the input dataset.
    zarr.copy_store(
        source.store,
        root.store,
        excludes=r"(^|/)\.(mods_done|vcf_done)$",
        if_exists="replace",
    )
    genotypes = root["call_genotype"]
    # Match replacement of call_genotype by the original xarray implementation.
    genotypes.attrs.put({"_ARRAY_DIMENSIONS": ["variants", "samples", "ploidy"]})
    num_sites, num_samples, _ = genotypes.shape
    variant_chunk, sample_chunk, _ = genotypes.chunks
    error_mask = _create_error_array(
        root, "call_genotype_error_mask", ["variants", "samples", "ploidy"], bool
    )
    error_count = _create_error_array(
        root, "variant_genotype_error_count", ["variants"], np.int64
    )
    phase_mask = _create_error_array(
        root, "call_genotype_phase", ["variants", "samples"], bool
    )
    switch_count = _create_error_array(
        root, "sample_phase_switch_count", ["samples"], float
    )
    mispol_mask = _create_error_array(
        root, "variant_mispolarisation_mask", ["variants"], bool
    )
    mispol_ancestral = _create_error_array(
        root,
        "variant_mispolarised_ancestral_state",
        ["variants"],
        root["variant_ancestral_state"].dtype,
        filters=root["variant_ancestral_state"].filters,
    )

    rng = np.random.default_rng(seed=seed)
    het_counts = np.zeros(num_samples, dtype=np.int64)
    error_df = None
    if geno_multiplier > 0:
        error_df = pd.read_csv(error_csv_path, index_col=0)
    for start in range(0, num_sites, variant_chunk):
        stop = min(start + variant_chunk, num_sites)
        G_in = source["call_genotype"][start:stop]
        G_out = G_in
        if geno_multiplier > 0:
            G_out = add_empirical_genotype_errors(
                G_in,
                rng,
                fetch_empirical_probs,
                df=error_df,
                geno_multiplier=geno_multiplier,
            )
        unchanged = G_out == G_in
        error_mask[start:stop] = unchanged
        error_count[start:stop] = np.sum(~unchanged, axis=(1, 2))
        genotypes[start:stop] = G_out
        if space_per_switch > 0:
            het_counts += np.sum(G_out.sum(axis=2) == 1, axis=0)
        else:
            phase_mask[start:stop] = False
        del G_in, G_out, unchanged

    phase_ser = 0.0
    if space_per_switch > 0:
        num_het_pairs = np.maximum(het_counts - 1, 0).sum()
        if num_het_pairs == 0:
            raise ValueError("Cannot simulate switches without heterozygote pairs")
        phase_ser = num_samples * sequence_length / (space_per_switch * num_het_pairs)
        if phase_ser > 1:
            raise ValueError(
                "space_per_switch is too small for the heterozygote density"
            )
    root.attrs.update(
        space_per_switch=float(space_per_switch), phase_ser=float(phase_ser)
    )

    # Keep whole diplotypes together so switch detection crosses variant chunks.
    if phase_ser > 0:
        for start in range(0, num_samples, sample_chunk):
            stop = min(start + sample_chunk, num_samples)
            G_in = genotypes[:, start:stop, :]
            G_out, phase, counts = add_phase_switch_errors(G_in, phase_ser, rng)
            genotypes[:, start:stop, :] = G_out
            phase_mask[:, start:stop] = phase
            switch_count[start:stop] = counts
            del G_in, G_out, phase, counts
    else:
        switch_count[:] = 0

    for start in range(0, num_sites, variant_chunk):
        stop = min(start + variant_chunk, num_sites)
        include_mispol, ancestral = unbiased_mispolarise(
            root["variant_allele"][start:stop],
            root["variant_ancestral_state"][start:stop],
            mispol_rate,
            rng,
            root["variant_singleton_mask"][start:stop],
        )
        mispol_mask[start:stop] = ~include_mispol
        mispol_ancestral[start:stop] = ancestral

    for name in ["contig_id", "sample_id"]:
        original = root[name]
        values = original[:].astype(str)
        attrs = dict(original.attrs)
        array = root.create_dataset(
            name,
            data=values,
            chunks=original.chunks,
            compressor=original.compressor,
            fill_value=None,
            overwrite=True,
        )
        array.attrs.update(attrs)
    zarr.consolidate_metadata(root.store)
    output_path.touch()
