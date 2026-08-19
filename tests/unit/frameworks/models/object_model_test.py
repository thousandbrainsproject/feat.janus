# Copyright 2025-2026 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import os
import tempfile
import unittest

import numpy as np
import torch

from tbp.monty.frameworks.models.object_model import (
    GraphObjectModel,
    GridObjectModel,
    GridTooSmallError,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import check_orthonormal
from tbp.monty.geometry import Rotation


class ObjectModelTest(unittest.TestCase):
    def setUp(self):
        self.dummy_locs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
        self.dummy_pv = np.eye(3)
        pv = self.dummy_pv.flatten()
        self.dummy_features = {
            "pose_vectors": np.vstack([pv, pv, pv, pv]),
            "pose_fully_defined": np.array([True, True, True, True]),
            "hsv": np.array(
                [
                    [0.1, 1, 1],
                    [0.0, 1, 1],
                    [0.9, 1, 1],
                    [0.8, 1, 1],
                ]
            ),
            "curvature": np.array([0, 1, 2, 1]),
        }
        self.dummy_delta_thresholds = {
            "distance": 0.5,
            "pose_vectors": np.ones(3),
            "hsv": [0.1, 1, 1],
            "curvature": 0.1,
        }

    def test_create_graph_object_model(self):
        model = GraphObjectModel("test_model")
        self.assertIsNotNone(model)
        self.assertIsNone(model._graph)
        # make sure __repr__ works
        self.assertIsInstance(repr(model), str)

    def test_create_grid_object_model(self):
        model = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=10
        )
        self.assertIsNotNone(model)
        self.assertIsNone(model._graph)
        # make sure __repr__ works
        self.assertIsInstance(repr(model), str)

    def test_can_build_graph_object_model(self):
        model = GraphObjectModel("test_model")
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
            k_n=None,
            graph_delta_thresholds=self.dummy_delta_thresholds,
        )
        self.assertEqual(model.num_nodes, 4, "graph model should have 4 nodes.")
        for feature in self.dummy_features:
            self.assertIn(
                feature, model.feature_mapping.keys(), f"{feature} not stored in model."
            )

    def test_can_build_grid_object_model(self):
        max_nodes, max_size, num_voxels_per_dim = 10, 10, 10
        model = GridObjectModel("test_model", max_nodes, max_size, num_voxels_per_dim)
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
        )
        self.assertEqual(model.num_nodes, 4, "graph model should have 4 nodes.")
        for feature in self.dummy_features:
            self.assertIn(
                feature, model.feature_mapping.keys(), f"{feature} not stored in model."
            )

    def test_apply_delta_thresholds_correctly(self):
        # Test distance.
        model = GraphObjectModel("test_model")
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
            k_n=None,
            graph_delta_thresholds={
                "distance": 2,
            },
        )
        self.assertEqual(
            model.num_nodes,
            1,
            "Graph model should have 1 node since no locations are more that 2cm away "
            f"from the first one. Model has {model.num_nodes} nodes.",
        )
        # Test pose_vectors.
        model = GraphObjectModel("test_model")
        features = copy.deepcopy(self.dummy_features)
        features["pose_vectors"][1] = np.array([0, 1, 0, 1, 0, 0, 0, 0, 1])
        model.build_model(
            self.dummy_locs,
            features,
            k_n=None,
            graph_delta_thresholds={
                "distance": 2,
                "pose_vectors": np.ones(3),
            },
        )
        self.assertEqual(
            model.num_nodes,
            2,
            "Graph model should have 2 nodes since no locations are more that 2cm away "
            "from the first one but pose vectors 0 and 1 differ enough. "
            f"Model has {model.num_nodes} nodes.",
        )
        # Test hsv.
        model = GraphObjectModel("test_model")
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
            k_n=None,
            graph_delta_thresholds={
                "distance": 2,
                "hsv": [0.1, 1, 1],
            },
        )
        self.assertEqual(
            model.num_nodes,
            2,
            "Graph model should have 2 nodes since no locations are more that 2cm away "
            "from the first one but hsv 0 (0.1) and 3 (0.8) differ enough. "
            f"Model has {model.num_nodes} nodes.",
        )
        # Test curvature.
        model = GraphObjectModel("test_model")
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
            k_n=None,
            graph_delta_thresholds={
                "distance": 2,
                "curvature": 1,
            },
        )
        self.assertEqual(
            model.num_nodes,
            2,
            "Graph model should have 2 nodes since no locations are more that 2cm away "
            "from the first one but curvature 0 (0) and 2 (2) differ enough. "
            f"Model has {model.num_nodes} nodes.",
        )

    def test_grid_locations_binned_correctly(self):
        # In test_can_build_grid_object_model each voxel is of size 1cm^3 so every
        # location gets its own voxel. Here, each voxel is of size 2cm^3 which means
        # that all locations should be binned into the same voxel and the location value
        # should be averaged.
        model = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=5
        )
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
        )
        self.assertEqual(
            model.num_nodes,
            1,
            f"graph model should have 1 node but has {model.num_nodes}.",
        )
        self.assertListEqual(
            list(model.pos[0]),
            [0.5, 0.5, 0],
            "location stored in model should be average of all locations "
            f"([0.5, 0.5, 0]) but is {model.pos}.",
        )

    def test_grid_features_are_averaged_correctly(self):
        model = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=5
        )
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
        )
        self.assertListEqual(
            list(model.get_values_for_feature("pose_vectors")[0]),
            list(self.dummy_pv.flatten()),
            "pose_vectors not averaged correctly. "
            "Since all PVs are the same, average should be same as dummy_pv.",
        )
        self.assertEqual(
            model.get_values_for_feature("pose_fully_defined")[0],
            True,
            "Average of pose_fully_defined should be True (majority vote).",
        )
        avg_hsv = model.get_values_for_feature("hsv")[0]
        self.assertListEqual(
            list(avg_hsv),
            [0.95, 1, 1],
            "hsv not averaged correctly. "
            "(keep in mind we want the circular average where 0==1).",
        )
        self.assertEqual(
            model.get_values_for_feature("curvature")[0],
            np.mean(self.dummy_features["curvature"]),
            "Curvature not averaged correctly.",
        )
        # Check different cases to make sure pose_fully_defined is averaged correctly.
        features = copy.deepcopy(self.dummy_features)
        features["pose_fully_defined"] = np.array([False, False, False, True])
        model2 = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=5
        )
        model2.build_model(
            self.dummy_locs,
            features,
        )
        self.assertEqual(
            model2.get_values_for_feature("pose_fully_defined")[0],
            False,
            "Average of pose_fully_defined should be False (majority vote).",
        )

        features["pose_fully_defined"][0] = True
        model2.build_model(
            self.dummy_locs,
            features,
        )
        self.assertEqual(
            model2.get_values_for_feature("pose_fully_defined")[0],
            True,
            "Average of pose_fully_defined should be True (True in case of tie).",
        )
        # Check different cases to make sure pose_vectors are averaged correctly.
        #     Check opposite surface side vectors are sorted out.
        features = copy.deepcopy(self.dummy_features)
        rot = Rotation.from_euler("z", 180, degrees=True)
        opposite_pv = rot.apply(np.eye(3))
        features["pose_vectors"][0] = opposite_pv.flatten()
        model3 = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=5
        )
        model3.build_model(
            self.dummy_locs,
            features,
        )
        self.assertListEqual(
            list(np.round(model3.get_values_for_feature("pose_vectors")[0], 3)),
            list(self.dummy_pv.flatten()),
            "Average of pose_vectors should still be the same. Changed pv is ignored "
            "because its angle is too far from the other ones (other surface side)",
        )
        features["pose_vectors"][1] = opposite_pv.flatten()
        features["pose_vectors"][2] = opposite_pv.flatten()
        model3.build_model(
            self.dummy_locs,
            features,
        )
        self.assertListEqual(
            list(np.round(model3.get_values_for_feature("pose_vectors")[0], 3)),
            list(np.round(opposite_pv.flatten(), 3)),
            "If majority of pose_vectors are on the opposite side of the surface (here)"
            " opposite_pv), those should become the average.",
        )
        #    Check averaged pose vector are still unit vectors.
        features = copy.deepcopy(self.dummy_features)
        rot = Rotation.from_euler("z", 20, degrees=True)
        rotated_pv = rot.apply(np.eye(3))
        features["pose_vectors"][0] = rotated_pv.flatten()
        model4 = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=5
        )
        model4.build_model(
            self.dummy_locs,
            features,
        )
        avg_pvs = model4.get_values_for_feature("pose_vectors")[0].reshape((3, 3))
        for pv in avg_pvs:
            self.assertAlmostEqual(
                np.linalg.norm(pv), 1.0, 3, "Average PVs are not unit vectors anymore"
            )
        self.assertTrue(check_orthonormal(avg_pvs), "Average PVs are not orthonormal")

    def test_max_nodes_applied_correctly(self):
        model = GridObjectModel(
            "test_model", max_nodes=3, max_size=10, num_voxels_per_dim=10
        )
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
        )
        self.assertEqual(model.num_nodes, 3, "Max nodes not applied correctly.")

    def test_max_size_applied_correctly(self):
        """Test that GridTooSmallError is raised if locations are outside of grid."""
        model = GridObjectModel(
            "test_model", max_nodes=10, max_size=1, num_voxels_per_dim=10
        )
        with self.assertRaises(GridTooSmallError):
            model.build_model(
                self.dummy_locs,
                self.dummy_features,
            )

    def build_grid_model(self):
        model = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=10
        )
        model.build_model(
            self.dummy_locs,
            self.dummy_features,
        )
        return model

    def test_match_evidence_is_unannotated_by_default(self):
        model = self.build_grid_model()
        evidence_means, counts = model.get_match_evidence()
        self.assertEqual(len(evidence_means), model.num_nodes)
        self.assertTrue(
            np.all(np.isnan(evidence_means)),
            "No match evidence should be stored before annotating.",
        )
        self.assertTrue(
            np.all(counts == 0),
            "No annotation counts should be stored before annotating.",
        )

    def test_match_evidence_accumulates_as_exponential_moving_average(self):
        model = self.build_grid_model()
        model.annotate_match_evidence(node_ids=[0, 1], evidence_values=[1.0, -0.5])
        model.annotate_match_evidence(node_ids=[0], evidence_values=[-0.25])

        evidence_means, counts = model.get_match_evidence()
        smoothing = model.match_evidence_smoothing
        self.assertAlmostEqual(
            evidence_means[0],
            1.0 + smoothing * (-0.25 - 1.0),
            msg="Evidence for node 0 should be the EMA of [1.0, -0.25], seeded "
            "with the first value.",
        )
        self.assertEqual(counts[0], 2, "Node 0 was annotated twice.")
        self.assertAlmostEqual(
            evidence_means[1],
            -0.5,
            msg="Evidence for node 1 should be -0.5 (negative evidence allowed).",
        )
        self.assertEqual(counts[1], 1, "Node 1 was annotated once.")
        self.assertTrue(
            np.all(np.isnan(evidence_means[2:])),
            "Nodes that were never annotated should stay unannotated.",
        )

    def test_match_evidence_stays_bounded(self):
        """The moving average stays within the range of annotated values."""
        model = self.build_grid_model()
        for _ in range(100):
            model.annotate_match_evidence(node_ids=[0], evidence_values=[2.0])
        evidence_means, counts = model.get_match_evidence()
        self.assertAlmostEqual(
            evidence_means[0],
            2.0,
            msg="Repeatedly annotating with the same value should keep the "
            "average at that value instead of growing without bound.",
        )
        self.assertEqual(counts[0], 100)

    def test_match_evidence_weights_recent_values_more(self):
        """Old annotations decay so the EMA tracks recent match quality."""
        model = self.build_grid_model()
        # Node initially matched well...
        for _ in range(10):
            model.annotate_match_evidence(node_ids=[0], evidence_values=[2.0])
        # ...but recently matched poorly.
        for _ in range(50):
            model.annotate_match_evidence(node_ids=[0], evidence_values=[-1.0])
        evidence_means, _ = model.get_match_evidence()
        self.assertLess(
            evidence_means[0],
            -0.9,
            "After many recent negative matches, the EMA should have decayed "
            "close to the recent value despite the earlier positive matches.",
        )

    def test_match_evidence_survives_model_update(self):
        model = self.build_grid_model()
        model.annotate_match_evidence(node_ids=[0], evidence_values=[1.5])
        annotated_voxels = copy.deepcopy(model._match_evidence)

        # Rebuild the graph from the grids by adding the same observations again
        # (identity transform keeps them at the same locations/voxels).
        model.update_model(
            locations=self.dummy_locs,
            features=self.dummy_features,
            location_rel_model=np.zeros(3),
            object_location_rel_body=np.zeros(3),
            object_rotation=Rotation.from_euler("xyz", [0, 0, 0]),
        )

        self.assertEqual(
            model._match_evidence,
            annotated_voxels,
            "Voxel-level match evidence should be untouched by a model update.",
        )
        evidence_means, counts = model.get_match_evidence()
        annotated = ~np.isnan(evidence_means)
        self.assertEqual(
            np.sum(annotated),
            1,
            "Exactly one node should map to the annotated voxel after the "
            "graph rebuild.",
        )
        self.assertAlmostEqual(float(evidence_means[annotated][0]), 1.5)
        self.assertEqual(counts[annotated][0], 1)

    def test_match_evidence_survives_serialization_round_trip(self):
        model = self.build_grid_model()
        model.annotate_match_evidence(node_ids=[0, 2], evidence_values=[2.0, -1.0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.pt")
            torch.save(model, model_path)
            loaded_model = torch.load(model_path, weights_only=False)

        self.assertEqual(
            loaded_model._match_evidence,
            model._match_evidence,
            "Match evidence should persist through torch.save/torch.load.",
        )
        original_means, original_counts = model.get_match_evidence()
        loaded_means, loaded_counts = loaded_model.get_match_evidence()
        np.testing.assert_array_equal(loaded_means, original_means)
        np.testing.assert_array_equal(loaded_counts, original_counts)

    def test_match_evidence_works_with_original_graph_models(self):
        """Models loaded with use_original_graph=True have no grids.

        The location->voxel mapping needed for annotation should be initialized
        lazily in that case.
        """
        source_model = self.build_grid_model()
        model = GridObjectModel(
            "test_model", max_nodes=10, max_size=10, num_voxels_per_dim=10
        )
        model.use_original_graph = True
        model.set_graph(source_model._graph)

        model.annotate_match_evidence(node_ids=[1], evidence_values=[0.5])
        evidence_means, counts = model.get_match_evidence()
        self.assertAlmostEqual(float(evidence_means[1]), 0.5)
        self.assertEqual(counts[1], 1)
        annotated = ~np.isnan(evidence_means)
        self.assertEqual(np.sum(annotated), 1)
