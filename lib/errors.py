import numpy as np
import pandas as pd
import sgkit
import xarray as xr
from numba import njit


def scale_genotype_error_probs(probs, geno_multiplier):
    """Scale off-diagonal genotype-error probabilities independently by row."""
    if not np.isfinite(geno_multiplier) or geno_multiplier < 0:
        message = "geno_multiplier must be a finite value greater than or equal to 0"
        raise ValueError(message)

    scaled_probs = np.zeros_like(probs, dtype=float)
    for true_genotype in range(3):
        error_genotypes = np.arange(3) != true_genotype
        scaled_errors = probs[true_genotype, error_genotypes] * geno_multiplier
        off_diag_sum = scaled_errors.sum()
        if off_diag_sum >= 1:
            normalization_constant = 1 / off_diag_sum
            scaled_errors *= normalization_constant
            diagonal = 0
        else:
            diagonal = 1 - off_diag_sum
        scaled_probs[true_genotype, error_genotypes] = scaled_errors
        scaled_probs[true_genotype, true_genotype] = diagonal

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
    include_het_pairs = include_het[:-1] & include_het[1:]
    het_pairs_idx = include_het_pairs.nonzero()[0]

    include_switch_sites = rng.random(len(het_pairs_idx)) < phase_ser
    num_switches = np.sum(include_switch_sites)
    phase_array = np.zeros(d_in.shape[0], dtype=bool)
    if num_switches > 0:
        # select rightmost site in each pair
        switch_sites = het_pairs_idx[include_switch_sites] + 1
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
    The switch error rate (SER) is the probability of a single phase switch
    per adjacent heterozygous site pair.
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


def add_errors(
    ds,
    output_path,
    error_csv_path,
    geno_multiplier,
    phase_ser,
    mispol_rate,
    seed,
):
    def add_xarray(dict, array, dims, name):
        xarray = xr.DataArray(array, dims=dims, name=name)
        dict[name] = xarray

    new_vars = {}
    rng = np.random.default_rng(seed=seed)
    G_in = ds.call_genotype.values

    if not np.isfinite(geno_multiplier) or geno_multiplier < 0:
        message = "geno_multiplier must be a finite value greater than or equal to 0"
        raise ValueError(message)
    if not 0 <= phase_ser <= 1:
        raise ValueError("phase_ser must be between 0 and 1")
    if not 0 <= mispol_rate <= 1:
        raise ValueError("mispol_rate must be between 0 and 1")

    # Genotype errors
    if geno_multiplier > 0:
        error_df = pd.read_csv(error_csv_path, index_col=0)
        G_geno_error = add_empirical_genotype_errors(
            G_in,
            rng,
            fetch_empirical_probs,
            df=error_df,
            geno_multiplier=geno_multiplier,
        )
    else:
        G_geno_error = G_in
    error_mask = G_geno_error == G_in
    genotype_error_count = np.sum(~error_mask, axis=(1, 2))
    add_xarray(
        new_vars,
        error_mask,
        dims=["variants", "samples", "ploidy"],
        name="call_genotype_error_mask",
    )
    add_xarray(
        new_vars,
        genotype_error_count,
        dims=["variants"],
        name="variant_genotype_error_count",
    )

    # Phasing errors
    G_out, call_genotype_phase, sample_switch_count = add_phase_switch_errors(
        G_geno_error, phase_ser, rng
    )
    add_xarray(
        new_vars,
        call_genotype_phase,
        dims=["variants", "samples"],
        name="call_genotype_phase",
    )
    add_xarray(
        new_vars, sample_switch_count, dims=["samples"], name="sample_phase_switch_count"
    )

    # Mispolarisation errors
    variant_allele = ds.variant_allele.values
    ancestral_state = ds.variant_ancestral_state.values
    singleton_mask = ds.variant_singleton_mask.values
    include_mispol, mispol_ancestral = unbiased_mispolarise(
        variant_allele,
        ancestral_state,
        mispol_rate,
        rng,
        singleton_mask,
    )
    add_xarray(
        new_vars, ~include_mispol, dims=["variants"], name="variant_mispolarisation_mask"
    )
    add_xarray(
        new_vars,
        mispol_ancestral,
        dims=["variants"],
        name="variant_mispolarised_ancestral_state",
    )

    new_ds = ds.copy()
    v_chunk = ds.call_genotype.chunks[0][0]
    s_chunk = ds.call_genotype.chunks[1][0]
    G_xr = xr.DataArray(
        G_out, dims=["variants", "samples", "ploidy"], name="call_genotype"
    ).chunk({"variants": v_chunk, "samples": s_chunk})
    new_ds["call_genotype"] = G_xr
    new_ds.update(new_vars)
    sgkit.save_dataset(new_ds, output_path.parent)
    output_path.touch()
