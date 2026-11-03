from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sgkit
import xarray as xr
import zarr
from numpy.testing import assert_array_equal

from lib import errors


@pytest.fixture
def rng():
    return np.random.default_rng(21)


@pytest.fixture
def small_genotype_matrix(seed=21):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(12, 12, 2), dtype=np.int8)


@pytest.fixture
def large_genotype_matrix(seed=21):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(1000, 1000, 2), dtype=np.int8)


class TestGenotypeErrorsFixedProbs:
    """
    Test genotype error simulations when the probability matrix is fixed.
    """

    @pytest.fixture(scope="class")
    def identity_probs(self):
        return np.eye(4, dtype=float)

    @pytest.fixture(scope="class")
    def to_zero_probs(self):
        P = np.zeros((4, 4), dtype=float)
        P[:, 0] = 1.0
        return P

    @staticmethod
    def _apply_site_samplewise(G_site, probs, rng):
        out = np.empty_like(G_site)
        for i, g in enumerate(G_site):
            out[i] = errors.sample_genotype(g, probs, rng)
        return out

    def test_identity_matrix_fixes_genotypes(
        self, small_genotype_matrix, identity_probs, rng
    ):
        G_in = small_genotype_matrix
        G_out = np.full_like(G_in, 0, dtype=np.int8)
        for site in range(G_in.shape[0]):
            g_in = G_in[site]
            g_samplewise = self._apply_site_samplewise(g_in, identity_probs, rng)
            g_vectorised = errors.sample_genotypes_vectorised(g_in, identity_probs, rng)
            assert_array_equal(g_samplewise, g_in)
            assert_array_equal(g_vectorised, g_in)
            G_out[site] = g_in

        def probs_func(freq):
            return identity_probs

        G_out_full = errors.add_empirical_genotype_errors(G_in, rng, probs_func)
        assert_array_equal(G_out, G_out_full)

    def test_to_zero_matrix_fixes_genotypes(
        self, small_genotype_matrix, to_zero_probs, rng
    ):
        G_in = small_genotype_matrix
        G_out = np.full_like(G_in, 0, dtype=np.int8)
        for site in range(G_in.shape[0]):
            g_in = G_in[site]
            g_samplewise = self._apply_site_samplewise(g_in, to_zero_probs, rng)
            g_vectorised = errors.sample_genotypes_vectorised(g_in, to_zero_probs, rng)
            assert_array_equal(g_samplewise, G_out[site])
            assert_array_equal(g_vectorised, G_out[site])
            G_out[site] = 0

        def probs_func(freq):
            return to_zero_probs

        G_out_full = errors.add_empirical_genotype_errors(G_in, rng, probs_func)
        assert_array_equal(G_out, G_out_full)


class TestFetchEmpiricalProbs:
    """
    Check behaviour of ``fetch_empirical_probs`` using the empirical error dataset.
    """

    @pytest.fixture(scope="class")
    def fetch_df(self):
        path = Path("data/external/EmpiricalErrorPlatinum1000G.csv")
        if not path.exists():
            pytest.skip(f"Empirical error table not found at {path!s}")
        return pd.read_csv(path, index_col=0)

    @pytest.mark.parametrize("freq", [0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_prob_matrix_rows_sum_to_one(self, freq, fetch_df):
        probs = errors.fetch_empirical_probs(freq, fetch_df)
        print(probs)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_extreme_frequency_maps_to_hom_alt(self, fetch_df, rng):
        probs = errors.fetch_empirical_probs(1.0, fetch_df)
        assert np.allclose(probs[:, 3], 1.0)
        assert np.allclose(probs[:, :3], 0.0)

        G = rng.integers(0, 2, size=(3, 4, 2), dtype=np.int8)
        out = errors.add_empirical_genotype_errors(G, rng, lambda f: probs)
        assert np.all(out == 1)


class TestGenotypeErrorMultiplier:
    @pytest.fixture
    def genotype_probs(self):
        return np.array(
            [
                [0.90, 0.08, 0.02],
                [0.05, 0.90, 0.05],
                [0.02, 0.08, 0.90],
            ]
        )

    def test_zero_gives_identity(self, genotype_probs):
        scaled = errors.scale_genotype_error_probs(genotype_probs, 0)
        assert np.allclose(scaled, np.eye(3))

    def test_one_preserves_matrix(self, genotype_probs):
        scaled = errors.scale_genotype_error_probs(genotype_probs, 1)
        assert np.allclose(scaled, genotype_probs)

    def test_zero_empirical_error_remains_error_free(self):
        scaled = errors.scale_genotype_error_probs(np.eye(3), 2)
        assert np.allclose(scaled, np.eye(3))

    def test_two_scales_error_hazard(self, genotype_probs):
        scaled = errors.scale_genotype_error_probs(genotype_probs, 2)
        expected = np.array(
            [
                [0.81, 0.152, 0.038],
                [0.095, 0.81, 0.095],
                [0.038, 0.152, 0.81],
            ]
        )
        assert np.allclose(scaled, expected)

    def test_scaling_preserves_error_proportions(self):
        probs = np.array(
            [
                [0.40, 0.50, 0.10],
                [0.05, 0.90, 0.05],
                [0.02, 0.08, 0.90],
            ]
        )
        scaled = errors.scale_genotype_error_probs(probs, 2)
        assert np.allclose(scaled[0], [0.16, 0.70, 0.14])
        assert np.isclose(scaled[0, 1] / scaled[0, 2], 5)

    def test_finite_multiplier_does_not_saturate(self):
        probs = np.array(
            [
                [0.50, 0.40, 0.10],
                [0.05, 0.90, 0.05],
                [0.02, 0.08, 0.90],
            ]
        )
        scaled = errors.scale_genotype_error_probs(probs, 2)
        assert np.allclose(scaled[0], [0.25, 0.60, 0.15])

    def test_negative_multiplier_rejected(self, genotype_probs):
        with pytest.raises(ValueError):
            errors.scale_genotype_error_probs(genotype_probs, -1)


class TestEncodeDecodeRoundTrip:
    """
    Round‑tripping through encode/ decode must be lossless.
    """

    @pytest.mark.parametrize(
        "gt", [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
    )
    def test_single(self, gt):
        idx = errors.encode_genotypes(np.array([gt]))[0]
        assert_array_equal(errors.decode_genotypes(idx), gt)

    def test_vectorised(self, rng):
        G = rng.integers(0, 2, size=(20, 2), dtype=np.int8)
        assert_array_equal(errors.decode_genotypes(errors.encode_genotypes(G)), G)


class TestSinglePhaseSwitch:
    """Test Numba phase switch function."""

    @pytest.fixture(scope="class")
    def sample_diplotypes(self):
        rng = np.random.default_rng(7)
        D_random = rng.integers(0, 2, size=(8, 2), dtype=np.int8)
        D_zero_one = np.tile(np.array([[0, 1]], dtype=np.int8), (8, 1))
        D_one_zero = np.tile(np.array([[1, 0]], dtype=np.int8), (8, 1))
        return D_random, D_zero_one, D_one_zero

    @pytest.mark.parametrize("switch_site", [0, 1, 4, 8])
    def test_round_trip(self, sample_diplotypes, switch_site):
        switch_sites = np.array([switch_site])
        for d_in in sample_diplotypes:
            d_out = np.zeros_like(d_in)
            phase_array = np.zeros(d_in.shape[0], dtype=bool)
            d_out1, phase_array1 = errors.phase_switch_diplotype(
                d_in, d_out, phase_array, switch_sites
            )
            d_out2, phase_array2 = errors.phase_switch_diplotype(
                d_out1, d_out, phase_array1, switch_sites
            )
            assert_array_equal(d_out2, d_in)
            assert_array_equal(phase_array2, phase_array)

    def test_zero_one_flips(self, num_sites=8):
        d_in = np.tile(np.array([[0, 1]], dtype=np.int8), (num_sites, 1))
        expected = np.tile(np.array([[1, 0]], dtype=np.int8), (num_sites, 1))
        d_out = np.zeros_like(d_in)
        phase_array = np.zeros(d_in.shape[0], dtype=bool)
        switch_sites = np.array([0])
        d_out, phase_array = errors.phase_switch_diplotype(
            d_in, d_out, phase_array, switch_sites
        )
        assert_array_equal(d_out, expected)
        assert_array_equal(phase_array, np.ones(8, dtype=bool))


class TestPhaseSwitchErrorRate:
    """Tests whether phase switch sampling of diplotypes based on SER."""

    def test_zero_error_rate(self, large_genotype_matrix, rng):
        G_in = large_genotype_matrix
        G_out, call_genotype_phase, sample_switch_count = errors.add_phase_switch_errors(
            G_in, phase_ser=0, rng=rng
        )
        num_sites = G_in.shape[0]
        num_samples = G_in.shape[1]
        zero_genotype_phase = np.zeros([num_sites, num_samples], dtype=bool)
        zero_switch_count = np.zeros(num_samples)
        assert_array_equal(G_out, G_in)
        assert_array_equal(call_genotype_phase, zero_genotype_phase)
        assert_array_equal(sample_switch_count, zero_switch_count)

    @pytest.mark.parametrize("spacing", [1, 2, 4])
    def test_maximum_error_rate(self, rng, spacing):
        """
        At SER 1, switch at every heterozygote after the first,
        regardless of intervening homozygous sites.
        """
        d_in = np.zeros((4 * spacing, 2), dtype=np.int8)
        d_in[::spacing] = [0, 1]
        d_expected = d_in.copy()
        d_expected[spacing :: 2 * spacing] = [1, 0]
        phase_expected = np.repeat([False, True, False, True], spacing)
        d_out, phase_array, num_switches = errors.sample_phase_switches(
            d_in, phase_ser=1, rng=rng
        )
        assert_array_equal(d_out, d_expected)
        assert_array_equal(phase_array, phase_expected)
        assert num_switches == 3

    @pytest.mark.parametrize("het_sites", [[], [0], [2], [4]])
    def test_fewer_than_two_heterozygotes(self, rng, het_sites):
        d_in = np.zeros((5, 2), dtype=np.int8)
        d_in[het_sites] = [0, 1]
        d_out, phase_array, num_switches = errors.sample_phase_switches(
            d_in, phase_ser=1, rng=rng
        )
        assert_array_equal(d_out, d_in)
        assert_array_equal(phase_array, np.zeros(5, dtype=bool))
        assert num_switches == 0


class TestInvalidGenotypeData:
    """
    Test that invalid genotype data raises appropriate errors.
    """

    def test_multiallelic_not_allowed(self, rng):
        G = np.array([[[0, 1], [2, 2]]], dtype=np.int8)
        with pytest.raises(AssertionError):
            errors.add_empirical_genotype_errors(G, rng, lambda f: np.eye(4))
        with pytest.raises(AssertionError):
            errors.add_phase_switch_errors(G, phase_ser=0, rng=rng)

    def test_wrong_ploidy(self, rng):
        G = np.zeros((2, 3, 3), dtype=np.int8)
        with pytest.raises(AssertionError):
            errors.add_empirical_genotype_errors(G, rng, lambda f: np.eye(4))
        with pytest.raises(AssertionError):
            errors.add_phase_switch_errors(G, phase_ser=0, rng=rng)

    def test_wrong_rank(self, rng):
        G = np.zeros((4, 2), dtype=np.int8)
        with pytest.raises(AssertionError):
            errors.add_empirical_genotype_errors(G, rng, lambda f: np.eye(4))
        with pytest.raises(AssertionError):
            errors.add_phase_switch_errors(G, phase_ser=0, rng=rng)


class TestUnbiasedMispolarise:
    """
    Basic tests for unbiased mispolarisation error simulation.
    """

    @pytest.fixture(scope="class")
    def four_allele_example(self):
        variant_allele = np.tile(np.array([["A", "T"]], dtype="<U1"), (4, 1))
        ancestral_state = np.full(4, "A", dtype="<U1")
        return variant_allele, ancestral_state

    @pytest.fixture(scope="class")
    def singleton_mask(self):
        return np.array([True, False, False, True], dtype=bool)

    def test_incorrect_ancestral_state(self, four_allele_example, rng):
        variant_allele, _ = four_allele_example
        wrong_ancestral_state = np.full(4, "T", dtype="<U1")
        with pytest.raises(AssertionError):
            errors.unbiased_mispolarise(
                variant_allele,
                wrong_ancestral_state,
                0.5,
                rng,
                singleton_mask=np.zeros(4, dtype=bool),
            )

    def test_zero_mispol_rate(self, four_allele_example, singleton_mask, rng):
        variant_allele, ancestral_state = four_allele_example
        include_mispol, mispol_ancestral = errors.unbiased_mispolarise(
            variant_allele,
            ancestral_state,
            mispol_rate=0,
            rng=rng,
            singleton_mask=singleton_mask,
        )
        assert_array_equal(include_mispol, np.zeros(4, dtype=bool))
        assert_array_equal(mispol_ancestral, ancestral_state)

    def test_maximum_mispol_rate(self, four_allele_example, rng):
        variant_allele, ancestral_state = four_allele_example
        include_mispol, mispol_ancestral = errors.unbiased_mispolarise(
            variant_allele,
            ancestral_state,
            mispol_rate=1,
            rng=rng,
            singleton_mask=np.zeros(4, dtype=bool),
        )
        assert_array_equal(include_mispol, np.ones(4, dtype=bool))
        assert_array_equal(mispol_ancestral, np.full(4, "T", dtype="<U1"))


class TestChunkedErrorAddition:
    @pytest.fixture
    def dataset(self):
        rng = np.random.default_rng(123)
        genotypes = rng.integers(0, 2, size=(17, 7, 2), dtype=np.int8)
        genotypes[0] = 0
        genotypes[-1] = 1
        genotypes[:, 0] = [0, 1]
        genotypes[:, 1] = 0
        allele_count = genotypes.sum(axis=(1, 2))
        dataset = xr.Dataset(
            {
                "call_genotype": (("variants", "samples", "ploidy"), genotypes),
                "call_genotype_mask": (("variants", "samples", "ploidy"), genotypes < 0),
                "variant_position": ("variants", np.arange(17)),
                "variant_allele": (
                    ("variants", "alleles"),
                    np.tile(np.array(["A", "T"], dtype=object), (17, 1)),
                ),
                "variant_ancestral_state": ("variants", np.full(17, "A", dtype=object)),
                "variant_singleton_mask": ("variants", allele_count == 1),
                "variant_allele_count": ("variants", allele_count),
                "sample_id": ("samples", np.arange(7).astype(str).astype(object)),
                "contig_id": ("contigs", np.array(["chr20"], dtype=object)),
            },
            attrs={"description": "error addition regression fixture"},
        )
        dataset.call_genotype.attrs["description"] = "Genotypes before errors"
        return dataset

    @pytest.fixture
    def error_csv_path(self):
        return Path("data/external/EmpiricalErrorPlatinum1000G.csv")

    def in_memory_output(self, ds, error_csv_path, geno, space_per_switch, mispol, seed):
        """Calculate the expected error output in memory for a 17 bp interval."""
        ds = ds.copy(deep=True)
        rng = np.random.default_rng(seed)
        G_in = ds.call_genotype.values
        G_geno_error = G_in
        if geno > 0:
            df = pd.read_csv(error_csv_path, index_col=0)
            G_geno_error = errors.add_empirical_genotype_errors(
                G_in, rng, errors.fetch_empirical_probs, df=df, geno_multiplier=geno
            )
        unchanged = G_geno_error == G_in
        counts = np.sum(~unchanged, axis=(1, 2))
        phase_ser = 0.0
        if space_per_switch > 0:
            het_counts = np.sum(G_geno_error.sum(axis=2) == 1, axis=0)
            num_het_pairs = np.maximum(het_counts - 1, 0).sum()
            phase_ser = 7 * 17 / (space_per_switch * num_het_pairs)
        ds.attrs.update(space_per_switch=float(space_per_switch), phase_ser=phase_ser)
        G_out, phase_mask, switches = errors.add_phase_switch_errors(
            G_geno_error, phase_ser, rng
        )
        include_mispol, ancestral = errors.unbiased_mispolarise(
            ds.variant_allele.values,
            ds.variant_ancestral_state.values,
            mispol,
            rng,
            ds.variant_singleton_mask.values,
        )
        ds["call_genotype"] = (("variants", "samples", "ploidy"), G_out)
        ds["call_genotype_error_mask"] = (("variants", "samples", "ploidy"), unchanged)
        ds["variant_genotype_error_count"] = ("variants", counts)
        ds["call_genotype_phase"] = (("variants", "samples"), phase_mask)
        ds["sample_phase_switch_count"] = ("samples", switches)
        ds["variant_mispolarisation_mask"] = ("variants", ~include_mispol)
        ds["variant_mispolarised_ancestral_state"] = ("variants", ancestral)
        for name in ["contig_id", "sample_id"]:
            ds[name] = ds[name].astype(str)
        return ds

    @pytest.mark.parametrize("chunks", [(1, 1), (4, 3), (8, 7), (17, 7)])
    @pytest.mark.parametrize(
        "geno,space_per_switch,mispol",
        [(0, 0, 0), (1, 0, 0.4), (1, 40, 0.4), (2, 20, 1), (0, 40, 0)],
    )
    @pytest.mark.parametrize("seed", [21, 42])
    def test_matches_in_memory(
        self,
        tmp_path,
        dataset,
        error_csv_path,
        chunks,
        geno,
        space_per_switch,
        mispol,
        seed,
    ):
        source_path = tmp_path / "source.zarr"
        output_path = tmp_path / "output.zarr" / ".mods_done"
        variant_chunk, sample_chunk = chunks
        source = dataset.chunk({"variants": variant_chunk, "samples": sample_chunk})
        sgkit.save_dataset(source, source_path)
        (source_path / ".mods_done").touch()
        (source_path / ".vcf_done").touch()
        expected = self.in_memory_output(
            dataset, error_csv_path, geno, space_per_switch, mispol, seed
        )
        errors.add_errors(
            source_path,
            output_path,
            error_csv_path,
            geno,
            space_per_switch,
            mispol,
            seed,
            sequence_length=17,
        )
        actual = sgkit.load_dataset(output_path.parent)
        xr.testing.assert_identical(actual.compute(), expected)
        xr.testing.assert_identical(sgkit.load_dataset(source_path).compute(), dataset)
        assert output_path.exists()
        assert not (output_path.parent / ".vcf_done").exists()
        root = zarr.open_group(str(output_path.parent), mode="r")
        assert root["call_genotype"].chunks == (variant_chunk, sample_chunk, 2)
        assert root["call_genotype_error_mask"].chunks == root["call_genotype"].chunks

    @pytest.mark.parametrize("space_per_switch", [0, 40])
    def test_genotype_reads_are_bounded(
        self, tmp_path, dataset, error_csv_path, monkeypatch, space_per_switch
    ):
        source_path = tmp_path / "source.zarr"
        output_path = tmp_path / "output.zarr" / ".mods_done"
        sgkit.save_dataset(dataset.chunk({"variants": 4, "samples": 3}), source_path)
        original_getitem = zarr.Array.__getitem__
        read_shapes = []

        def getitem(array, selection):
            result = original_getitem(array, selection)
            if array.name == "/call_genotype":
                read_shapes.append(result.shape)
                assert result.shape[0] <= 4 or result.shape[1] <= 3
            return result

        monkeypatch.setattr(zarr.Array, "__getitem__", getitem)
        errors.add_errors(
            source_path,
            output_path,
            error_csv_path,
            1,
            space_per_switch,
            0.4,
            21,
            sequence_length=17,
        )
        assert len(read_shapes) > 0
