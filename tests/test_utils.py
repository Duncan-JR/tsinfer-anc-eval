import io
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
import tskit
import xarray as xr
import yaml

config_path = pathlib.Path(__file__).resolve().parents[1] / "config.yaml"
config = yaml.safe_load(config_path.read_text())
tsinfer_path = pathlib.Path(config["tsinfer"][0]["path"]).expanduser()
sys.path.insert(0, str(tsinfer_path))

import tsinfer  # noqa: E402

from lib import utils  # noqa: E402


@pytest.fixture
def tree_sequence():
    tables = tskit.TableCollection(sequence_length=80)
    for time in [0, 0, 2, 3, 4]:
        flags = tskit.NODE_IS_SAMPLE if time == 0 else 0
        tables.nodes.add_row(time=time, flags=flags)
    for left, right in [(0, 20), (30, 60), (70, 80)]:
        tables.edges.add_row(left, right, parent=2, child=0)
        tables.edges.add_row(left, right, parent=4, child=2)
    tables.edges.add_row(20, 80, parent=3, child=1)
    tables.edges.add_row(20, 80, parent=4, child=3)
    for position, node in [(10, 2), (15, 2), (20, 3), (30, 3)]:
        site = tables.sites.add_row(position, ancestral_state="A")
        tables.mutations.add_row(site, node=node, derived_state="T")
    site = tables.sites.add_row(40, ancestral_state="A")
    parent = tables.mutations.add_row(site, node=4, derived_state="T")
    tables.mutations.add_row(site, node=2, derived_state="C", parent=parent)
    for position, node in [(50, 2), (60, 3), (70, 0)]:
        site = tables.sites.add_row(position, ancestral_state="A")
        tables.mutations.add_row(site, node=node, derived_state="T")
    tables.sort()
    return tables.tree_sequence()


@pytest.fixture
def chunk_dataframe():
    return pd.DataFrame(
        {
            "inf_focal_site": [2, 3, 6],
            "true_focal_site": [3, 4, 7],
            "focal_position": [30, 40, 70],
            "inf_node": [1, 1, 3],
            "true_node": [3, 2, 2],
        }
    )


@pytest.fixture
def annotations(tree_sequence):
    return xr.Dataset(
        {
            "variant_position": ("variants", tree_sequence.sites_position),
            "variant_mispolarisation_mask": (
                "variants",
                [True, False, False, True, False, True, False, True],
            ),
            "variant_allele_frequency": (
                "variants",
                [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            ),
            "variant_genotype_error_count": ("variants", [1, 2, 3, 4, 5, 6, 7, 8]),
        },
        attrs={"space_per_switch": 2e7, "phase_ser": 0.01},
    )


@pytest.fixture
def ancestor_data(tmp_path):
    positions = [
        np.array([10, 20, 30, 40, 50, 60, 70]),
        np.array([10, 15, 20, 30, 40, 50, 60, 70]),
    ]
    ancestors = [
        [(0, 6, 0.8, [2, 3], [1, 0, 1, -1, 0, 1]), (2, 7, 0.4, [6], [0, 1, 0, 1, 1])],
        [(2, 8, 0.7, [3, 4], [1, 0, 1, 1, 0, 1]), (5, 8, 0.3, [7], [1, 0, 0])],
    ]
    stores = {}
    for version, sites, records in zip(["v1", "v2"], positions, ancestors):
        path = tmp_path / f"{version}.ancestors"
        with tsinfer.AncestorData(
            position=sites,
            sequence_length=80,
            path=str(path),
            chunk_size=2,
            chunk_size_sites=3,
        ) as data:
            data.add_ancestor(
                0, len(sites), 1.0, [], np.zeros(len(sites), dtype=np.int8)
            )
            data.add_ancestor(*records[0])
            data.add_ancestor(
                0, len(sites), 0.5, [], np.zeros(len(sites), dtype=np.int8)
            )
            data.add_ancestor(*records[1])
        stores[version] = data
    yield stores
    for data in stores.values():
        data.close()


class TestDecodeTrueGenotypes:
    @pytest.fixture
    def expanded_tree_sequence(self, tree_sequence):
        tables = tree_sequence.dump_tables()
        flags = tables.nodes.flags
        flags[:] = tskit.NODE_IS_SAMPLE
        tables.nodes.flags = flags
        return tables.tree_sequence()

    def test_compact_genotypes(self, expanded_tree_sequence):
        true_nodes = np.array([3, 2])
        true_shared_idx = np.array([0, 2, 3, 4, 5, 6, 7])
        actual = utils.decode_true_genotypes(
            expanded_tree_sequence, true_nodes, true_shared_idx
        )
        expected = np.array(
            [[-1, 1, 1, 1, 0, 1, 0], [1, -1, 0, 1, 1, -1, 0]], dtype=np.int8
        )
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.int8
        assert actual.flags.c_contiguous


class TestProcessAncestorChunk:
    expected_header = (
        "inferred_node,true_node,replicate,error_profile,geno_multiplier,"
        "space_per_switch,phase_ser,mispol_rate,version,side,inf_focal_site,true_focal_site,"
        "num_mutations,focal_position,focal_site_mispolarised,"
        "focal_site_genotype_error_count,allele_frequency,true_time,"
        "inferred_time,true_boundary,true_span,inferred_boundary,inferred_span,"
        "overlap_boundary,overlap_span,overlap_site_span,num_errors,"
        "num_errors_per_bp,num_errors_per_site,num_should_be_0,num_should_be_1,"
        "overshoot\n"
    )
    expected_csv = {
        1: """\
1,3,2,all,2.0,20000000.0,0.01,0.1,v1,left,2,3,1,30,False,4,0.3,3.0,0.8,20.0,51.0,10.0,60.0,20.0,50.0,5,2,0.04,0.4,0,0,10.0
1,3,2,all,2.0,20000000.0,0.01,0.1,v1,right,2,3,1,30,False,4,0.3,3.0,0.8,71.0,51.0,70.0,60.0,70.0,50.0,5,2,0.04,0.4,0,0,-1.0
1,2,2,all,2.0,20000000.0,0.01,0.1,v1,left,3,4,2,40,True,5,0.4,2.0,0.8,10.0,61.0,10.0,60.0,10.0,60.0,6,4,0.06666666666666667,0.6666666666666666,2,0,0.0
1,2,2,all,2.0,20000000.0,0.01,0.1,v1,right,3,4,2,40,True,5,0.4,2.0,0.8,71.0,61.0,70.0,60.0,70.0,60.0,6,4,0.06666666666666667,0.6666666666666666,2,0,-1.0
3,2,2,all,2.0,20000000.0,0.01,0.1,v1,left,6,7,1,70,False,8,0.7,2.0,0.4,10.0,61.0,30.0,41.0,30.0,41.0,5,3,0.07317073170731707,0.6,1,2,-20.0
3,2,2,all,2.0,20000000.0,0.01,0.1,v1,right,6,7,1,70,False,8,0.7,2.0,0.4,71.0,61.0,71.0,41.0,71.0,41.0,5,3,0.07317073170731707,0.6,1,2,0.0
""",
        2: """\
1,3,2,all,2.0,20000000.0,0.01,0.1,v1,left,2,3,1,30,False,4,0.3,3.0,0.8,20.0,51.0,10.0,60.0,20.0,50.0,5,2,0.04,0.4,0,0,10.0
1,3,2,all,2.0,20000000.0,0.01,0.1,v1,right,2,3,1,30,False,4,0.3,3.0,0.8,71.0,51.0,70.0,60.0,70.0,50.0,5,2,0.04,0.4,0,0,-1.0
1,3,2,all,2.0,20000000.0,0.01,0.1,v2,left,2,3,1,30,False,4,0.3,3.0,0.7,20.0,51.0,20.0,51.0,20.0,50.0,5,4,0.08,0.8,1,3,0.0
1,3,2,all,2.0,20000000.0,0.01,0.1,v2,right,2,3,1,30,False,4,0.3,3.0,0.7,71.0,51.0,71.0,51.0,70.0,50.0,5,4,0.08,0.8,1,3,0.0
1,2,2,all,2.0,20000000.0,0.01,0.1,v1,left,3,4,2,40,True,5,0.4,2.0,0.8,10.0,61.0,10.0,60.0,20.0,50.0,5,4,0.08,0.8,2,0,0.0
1,2,2,all,2.0,20000000.0,0.01,0.1,v1,right,3,4,2,40,True,5,0.4,2.0,0.8,71.0,61.0,70.0,60.0,70.0,50.0,5,4,0.08,0.8,2,0,-1.0
1,2,2,all,2.0,20000000.0,0.01,0.1,v2,left,3,4,2,40,True,5,0.4,2.0,0.7,10.0,61.0,20.0,51.0,20.0,50.0,5,2,0.04,0.4,1,1,-10.0
1,2,2,all,2.0,20000000.0,0.01,0.1,v2,right,3,4,2,40,True,5,0.4,2.0,0.7,71.0,61.0,71.0,51.0,70.0,50.0,5,2,0.04,0.4,1,1,0.0
3,2,2,all,2.0,20000000.0,0.01,0.1,v1,left,6,7,1,70,False,8,0.7,2.0,0.4,10.0,61.0,30.0,41.0,50.0,21.0,3,2,0.09523809523809523,0.6666666666666666,1,1,-20.0
3,2,2,all,2.0,20000000.0,0.01,0.1,v1,right,6,7,1,70,False,8,0.7,2.0,0.4,71.0,61.0,71.0,41.0,71.0,21.0,3,2,0.09523809523809523,0.6666666666666666,1,1,0.0
3,2,2,all,2.0,20000000.0,0.01,0.1,v2,left,6,7,1,70,False,8,0.7,2.0,0.3,10.0,61.0,50.0,21.0,50.0,21.0,3,1,0.047619047619047616,0.3333333333333333,1,0,-40.0
3,2,2,all,2.0,20000000.0,0.01,0.1,v2,right,6,7,1,70,False,8,0.7,2.0,0.3,71.0,61.0,71.0,21.0,71.0,21.0,3,1,0.047619047619047616,0.3333333333333333,1,0,0.0
""",
    }

    @pytest.mark.parametrize("num_versions", [1, 2])
    def test_complete_output(
        self,
        num_versions,
        tree_sequence,
        chunk_dataframe,
        annotations,
        ancestor_data,
        tmp_path,
    ):
        versions = ["v1", "v2"][:num_versions]
        anc_data_map = {version: ancestor_data[version] for version in versions}
        output_path = tmp_path / "chunk.csv"
        utils.process_ancestor_chunk(
            df=chunk_dataframe,
            ts=tree_sequence,
            ds=annotations,
            anc_data_map=anc_data_map,
            rep=2,
            error_profile="all",
            geno_multiplier=2.0,
            mispol_rate=0.1,
            output_path=output_path,
        )
        # Captured from the original processor and hand-checked: the terminal
        # boundary is 71, and missing inferred alleles keep their signed arithmetic.
        expected_csv = self.expected_header + self.expected_csv[num_versions]
        expected = pd.read_csv(io.StringIO(expected_csv))
        actual = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
