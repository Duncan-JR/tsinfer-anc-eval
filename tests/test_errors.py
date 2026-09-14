import numpy as np
import pandas as pd
import pytest
from pathlib import Path
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

    def test_identity_matrix_fixes_genotypes(self, small_genotype_matrix, identity_probs, rng):
        G_in = small_genotype_matrix
        G_out = np.full_like(G_in, 0, dtype=np.int8)
        for site in range(G_in.shape[0]):
            g_in = G_in[site]
            g_samplewise = self._apply_site_samplewise(g_in, identity_probs, rng)
            g_vectorised = errors.sample_genotypes_vectorised(g_in, identity_probs, rng)
            assert_array_equal(g_samplewise, g_in)
            assert_array_equal(g_vectorised, g_in)
            G_out[site] = g_in  
            
        probs_func = lambda freq: identity_probs
        G_out_full = errors.add_empirical_genotype_errors(G_in, rng, probs_func)
        assert_array_equal(G_out, G_out_full)

    def test_to_zero_matrix_fixes_genotypes(self, small_genotype_matrix, to_zero_probs, rng):
        G_in = small_genotype_matrix
        G_out = np.full_like(G_in, 0, dtype=np.int8)
        for site in range(G_in.shape[0]):
            g_in = G_in[site]
            g_samplewise = self._apply_site_samplewise(g_in, to_zero_probs, rng)
            g_vectorised = errors.sample_genotypes_vectorised(g_in, to_zero_probs, rng)
            assert_array_equal(g_samplewise, G_out[site])
            assert_array_equal(g_vectorised, G_out[site])
            G_out[site] = 0 

        probs_func = lambda freq: to_zero_probs
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


class TestEncodeDecodeRoundTrip:
    """
    Round‑tripping through encode/ decode must be lossless.
    """

    @pytest.mark.parametrize("gt", [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])])
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
            d_out1, phase_array1 = errors.phase_switch_diplotype(d_in, d_out, phase_array, switch_sites)
            d_out2, phase_array2 = errors.phase_switch_diplotype(d_out1, d_out, phase_array1, switch_sites)
            assert_array_equal(d_out2, d_in)
            assert_array_equal(phase_array2, phase_array)

    def test_zero_one_flips(self, num_sites=8):
        d_in = np.tile(np.array([[0, 1]], dtype=np.int8), (num_sites, 1))
        expected = np.tile(np.array([[1, 0]], dtype=np.int8), (num_sites, 1))
        d_out = np.zeros_like(d_in)
        phase_array = np.zeros(d_in.shape[0], dtype=bool)
        switch_sites=np.array([0])
        d_out, phase_array = errors.phase_switch_diplotype(d_in, d_out, phase_array, switch_sites)
        assert_array_equal(d_out, expected)
        assert_array_equal(phase_array, np.ones(8, dtype=bool))

class TestPhaseSwitchErrorRate:
    """Tests whether phase switch sampling of diplotypes based on SER."""

    def test_zero_error_rate(self, large_genotype_matrix, rng):
        G_in = large_genotype_matrix
        G_out, call_genotype_phase, sample_switch_count = errors.add_phase_switch_errors(G_in, switch_error_rate=0, rng=rng)
        num_sites = G_in.shape[0]
        num_samples = G_in.shape[1]
        zero_genotype_phase = np.zeros([num_sites, num_samples], dtype=bool)
        zero_switch_count = np.zeros(num_samples)
        assert_array_equal(G_out, G_in)
        assert_array_equal(call_genotype_phase, zero_genotype_phase)
        assert_array_equal(sample_switch_count, zero_switch_count)

    def test_maximum_error_rate(self, rng):
        """
        Given three consecutive pairs of het sites and an SER of 1,
        we should see the phase switch repeatedly after the first het
        site.
        """
        d_in = np.tile(np.array([[0, 1]], dtype=np.int8), (4, 1))
        d_expected = np.array([[0,1],[1,0],[0,1],[1,0]])
        phase_expected = np.array([0,1,0,1], dtype=bool)
        d_out, phase_array, num_switches = errors.sample_phase_switches(d_in, ser=1, rng=rng)
        assert_array_equal(d_out, d_expected)
        assert_array_equal(phase_array, phase_expected)
        assert num_switches == 3

class TestInvalidGenotypeData:
    """
    Test that invalid genotype data raises appropriate errors.
    """

    def test_multiallelic_not_allowed(self, rng):
        G = np.array([[[0, 1], [2, 2]]], dtype=np.int8)
        with pytest.raises(AssertionError):
            errors.add_empirical_genotype_errors(G, rng, lambda f: np.eye(4))
        with pytest.raises(AssertionError):
            errors.add_phase_switch_errors(G, switch_error_rate=0, rng=rng)

    def test_wrong_ploidy(self, rng):
        G = np.zeros((2, 3, 3), dtype=np.int8) 
        with pytest.raises(AssertionError):
            errors.add_empirical_genotype_errors(G, rng, lambda f: np.eye(4))
        with pytest.raises(AssertionError):
            errors.add_phase_switch_errors(G, switch_error_rate=0, rng=rng)

    def test_wrong_rank(self, rng):
        G = np.zeros((4, 2), dtype=np.int8)
        with pytest.raises(AssertionError):
            errors.add_empirical_genotype_errors(G, rng, lambda f: np.eye(4))
        with pytest.raises(AssertionError):
            errors.add_phase_switch_errors(G, switch_error_rate=0, rng=rng) 

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
