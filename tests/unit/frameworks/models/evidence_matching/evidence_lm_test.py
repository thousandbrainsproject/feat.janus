# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import copy
from unittest.mock import Mock, patch

import numpy as np
import quaternion as qt

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.goal_generation import (
    DenseExplorationGoalGenerator,
    EvidenceGoalGenerator,
)
from tbp.monty.frameworks.models.motor_policies import JumpToGoal
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    MotorSystemState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID
from tests.unit.resources.unit_test_utils import BaseGraphTest


class EvidenceLMTest(BaseGraphTest):
    def setUp(self):
        super().setUp()

        self.default_gsg_config = dict(
            elapsed_steps_factor=10,
            min_post_goal_success_steps=5,
            x_percent_scale_factor=0.75,
            desired_object_distance=0.03,
        )

    def get_elm_with_fake_object(
        self, fake_obs, initial_possible_poses="informed", gsg=None
    ):
        graph_lm = EvidenceGraphLM(
            max_match_distance=0.005,
            tolerances={
                "patch": {
                    "hsv": [0.1, 1, 1],
                    "principal_curvatures_log": [1, 1],
                }
            },
            feature_weights={
                "patch": {
                    "hsv": np.array([1, 0, 0]),
                }
            },
            # set graph size larger since fake obs displacements are meters
            max_graph_size=10,
            gsg=gsg,
            hypotheses_updater_args=dict(
                initial_possible_poses=initial_possible_poses,
            ),
        )
        graph_lm.mode = ExperimentMode.TRAIN
        for observation in fake_obs:
            graph_lm.exploratory_step(self.ctx, [observation])
        graph_lm.detected_object = "new_object0"
        graph_lm.detected_rotation_r = None
        graph_lm.buffer.stats["detected_location_rel_body"] = (
            graph_lm.buffer.get_current_location(input_channel="first")
        )

        self.assertEqual(
            len(graph_lm.buffer.get_all_locations_on_object(input_channel="first")),
            len(fake_obs),
            f"Should have stored exactly {fake_obs} locations in the buffer.",
        )
        graph_lm.update_ltm_from_stm()
        graph_lm.fixme_update_ground_truth()
        self.assertEqual(
            len(graph_lm.get_all_known_object_ids()),
            1,
            "Should have stored exactly one object in memory.",
        )
        self.assertEqual(
            graph_lm.get_all_known_object_ids()[0],
            "new_object0",
            "Learned object ID should be new_object0.",
        )
        return graph_lm

    def test_top_two_mlh_ids_without_current_hypotheses(self):
        """Return no MLH IDs before matching initializes current hypotheses."""
        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)
        graph_lm.reset_stm()

        self.assertEqual(graph_lm.get_top_two_mlh_ids(), (None, None))

    def get_elm_with_two_fake_objects(
        self,
        fake_obs,
        fake_obs_two,
        initial_possible_poses,
        gsg,
    ) -> EvidenceGraphLM:
        """Train on two fake observation objects.

        Returns:
            Evidence GraphLearning Module
        """
        # Train on first object
        graph_lm = self.get_elm_with_fake_object(
            fake_obs,
            initial_possible_poses=initial_possible_poses,
            gsg=gsg,
        )

        # Train on second object
        obj_two_target = copy.deepcopy(self.placeholder_target)
        obj_two_target["object"] = "new_object1"
        graph_lm.reset_stm()
        graph_lm.fixme_reset_ground_truth(primary_target=obj_two_target)
        for observation in fake_obs_two:
            graph_lm.exploratory_step(self.ctx, [observation])
        graph_lm.detected_object = obj_two_target["object"]
        graph_lm.detected_rotation_r = None
        graph_lm.buffer.stats["detected_location_rel_body"] = (
            graph_lm.buffer.get_current_location(input_channel="first")
        )

        self.assertEqual(
            len(graph_lm.buffer.get_all_locations_on_object(input_channel="first")),
            len(fake_obs_two),
            f"Should have stored exactly {fake_obs_two} locations in the buffer.",
        )
        graph_lm.update_ltm_from_stm()
        graph_lm.fixme_update_ground_truth()
        self.assertEqual(
            len(graph_lm.get_all_known_object_ids()),
            2,
            "Should have stored two objects in memory now.",
        )
        self.assertEqual(
            graph_lm.get_all_known_object_ids()[0],
            "new_object0",
            "Should still know first object.",
        )
        self.assertEqual(
            graph_lm.get_all_known_object_ids()[1],
            "new_object1",
            "Learned object ID should be new_object1.",
        )

        return graph_lm

    def test_symmetry_recognition(self):
        fake_obs_test = copy.deepcopy(self.fake_obs_symmetric)
        # Get LM with object learned from fake_obs
        graph_lm = self.get_elm_with_fake_object(self.fake_obs_symmetric)

        graph_lm.mode = ExperimentMode.EVAL
        # Don't need to give target object since we are not logging performance
        graph_lm.reset_stm()
        graph_lm.fixme_reset_ground_truth(primary_target=self.placeholder_target)
        num_steps_checked_symmetry = 0
        for i in range(12):
            observation = fake_obs_test[i % 4]
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])
            # since we don't have a monty class here we have to call this check
            # manually. Usually monty class coordinates terminal condition checks and
            # updates to symmetry count.
            graph_lm.get_unique_pose_if_available("new_object0")
            max_obj_evidence = np.max(graph_lm._hypotheses["new_object0"].evidence)
            if max_obj_evidence > graph_lm.object_evidence_threshold:
                num_steps_checked_symmetry += 1
                # On the first step we just store previous hypothesis ids.
                if num_steps_checked_symmetry > 1:
                    # On the second step we will add 1 symmetry evidence
                    # every step because we can't resolve between
                    # 0,0,0 and 180, 0, 180 rotation.
                    self.assertEqual(
                        graph_lm.symmetry_evidence,
                        num_steps_checked_symmetry - 1,
                        "Symmetry evidence doesn't seem to be as expected.",
                    )
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly one possible match.",
            )
            self.assertEqual(
                graph_lm.get_possible_matches()[0],
                "new_object0",
                "Should match to new_object0.",
            )
        self.assertListEqual(
            list(graph_lm.get_possible_poses()["new_object0"][0]),
            [0, 0, 0],
            "Should have rotation 0, 0, 0 as a possible pose.",
        )
        symmetry_pose = np.mod(
            np.array(graph_lm.get_possible_poses()["new_object0"][-1], dtype=float),
            360.0,
        )
        self.assertTrue(
            np.allclose(symmetry_pose, np.array([180.0, 0.0, 180.0])),
            "Since we have symmetry here, 180, 0, 180 should also be a possible pose.",
        )

    def test_match_evidence_annotation_after_confidence(self):
        """Model points are annotated once the hypothesis set has persisted.

        Before the persistence ("high confidence") condition is met, no match
        evidence should be written to the object model. Once the persistent
        hypotheses have narrowed down to a single object, every matching step
        should annotate the matched model points with the evidence they
        matched by.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)
        # Lower the persistence requirement so confidence is reached within a
        # few steps.
        graph_lm.required_symmetry_evidence = 3

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.reset_stm()
        graph_lm.fixme_reset_ground_truth(primary_target=self.placeholder_target)

        model = graph_lm.graph_memory.get_graph("new_object0", "patch")
        confidence_reached_at_step = None
        for step in range(12):
            observation = fake_obs_test[step % 4]
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])
            if confidence_reached_at_step is None:
                if graph_lm._persistent_hypothesis_ids:
                    confidence_reached_at_step = step
                else:
                    evidence_means, counts = model.get_match_evidence()
                    self.assertTrue(
                        np.all(np.isnan(evidence_means)) and np.all(counts == 0),
                        "Model should not be annotated before the persistence "
                        "condition is met.",
                    )

        self.assertIsNotNone(
            confidence_reached_at_step,
            "The LM should have reached the persistence condition.",
        )
        self.assertListEqual(
            list(graph_lm._persistent_hypothesis_ids.keys()),
            ["new_object0"],
            "Persistent hypotheses should be narrowed down to new_object0.",
        )
        evidence_means, counts = model.get_match_evidence()
        annotated = ~np.isnan(evidence_means)
        self.assertGreater(
            np.sum(annotated),
            0,
            "Model points matched by the persistent hypotheses should have "
            "been annotated.",
        )
        self.assertGreater(
            np.max(evidence_means[annotated]),
            0,
            "Since observations match the model exactly, at least one "
            "annotated point should have positive match evidence.",
        )
        self.assertTrue(
            np.all(counts[annotated] > 0),
            "Annotated points should have a positive annotation count.",
        )

    def test_same_sequence_recognition_elm(self):
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = ExperimentMode.EVAL
        # Don't need to give target object since we are not logging performance
        graph_lm.reset_stm()
        graph_lm.fixme_reset_ground_truth(primary_target=self.placeholder_target)
        target_evidence = 1
        for observation in fake_obs_test:
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly one possible match.",
            )
            self.assertEqual(
                graph_lm.get_possible_matches()[0],
                "new_object0",
                "Should match to new_object0.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["graph_id"],
                "new_object0",
                "new_object0 should be the mlh at every step.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["evidence"],
                target_evidence,
                "Since we have exact matches, evidence should be incremented by "
                "2 at each step (1 for pose match and 1 for feature match).",
            )
            target_evidence += 2

        self.assertListEqual(
            list(graph_lm.get_current_mlh()["rotation"].as_euler("xyz", degrees=True)),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_reverse_sequence_recognition_elm(self):
        """Test that object is recognized irrespective of sampling order."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test.reverse()

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.reset_stm()
        graph_lm.fixme_reset_ground_truth(primary_target=self.placeholder_target)
        target_evidence = 1
        for observation in fake_obs_test:
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly one possible match.",
            )
            self.assertEqual(
                graph_lm.get_possible_matches()[0],
                "new_object0",
                "Should match to new_object0.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["graph_id"],
                "new_object0",
                "new_object0 should be the mlh at every step.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["evidence"],
                target_evidence,
                "Since we have exact matches, evidence should be incremented by "
                "2 at each step (1 for pose match and 1 for feature match).",
            )
            target_evidence += 2

        self.assertListEqual(
            list(graph_lm.get_current_mlh()["rotation"].as_euler("xyz", degrees=True)),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_offset_sequence_recognition_elm(self):
        """Test that the object is recognized irrespective of its location rel body."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.reset_stm()
        graph_lm.fixme_reset_ground_truth(primary_target=self.placeholder_target)
        target_evidence = 1
        for observation in fake_obs_test:
            observation.location = observation.location + np.ones(3)
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly one possible match.",
            )
            self.assertEqual(
                graph_lm.get_possible_matches()[0],
                "new_object0",
                "Should match to new_object0.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["graph_id"],
                "new_object0",
                "new_object0 should be the mlh at every step.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["evidence"],
                target_evidence,
                "Since we have exact matches, evidence should be incremented by "
                "2 at each step (1 for pose match and 1 for feature match).",
            )
            target_evidence += 2

        self.assertListEqual(
            list(graph_lm.get_current_mlh()["rotation"].as_euler("xyz", degrees=True)),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_new_sampling_recognition_elm(self):
        """Test object recognition with slightly perturbed observations.

        Test that the object is recognized if observations don't exactly fit
        the model.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].location = fake_obs_test[0].location + np.ones(3) * 0.001
        fake_obs_test[2].location = fake_obs_test[2].location + np.ones(3) * 0.002

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.reset_stm()
        graph_lm.fixme_reset_ground_truth(primary_target=self.placeholder_target)
        for observation in fake_obs_test:
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly one possible match.",
            )
            self.assertEqual(
                graph_lm.get_possible_matches()[0],
                "new_object0",
                "Should match to new_object0.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["graph_id"],
                "new_object0",
                "new_object0 should be the mlh at every step.",
            )

        self.assertListEqual(
            list(graph_lm.get_current_mlh()["rotation"].as_euler("xyz", degrees=True)),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_different_locations_not_recognized_elm(self):
        """Test that the object is not recognized if locations don't match."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].location = fake_obs_test[0].location + np.ones(3) * 5

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.reset_stm()
        graph_lm.fixme_reset_ground_truth(primary_target=self.placeholder_target)
        for i, observation in enumerate(fake_obs_test):
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])
            if i < 2:
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    1,
                    "Should have one match at the first and second step.",
                )
            else:
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    0,
                    "Should have no possible matches.",
                )

    def _evaluate_target_location(
        self, graph_lm, fake_obs_test, target_object, focus_on_pose=False
    ):
        """Helper function for hypothesis testing that retreives a target location."""
        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.reset_stm()
        graph_lm.fixme_reset_ground_truth(primary_target=self.placeholder_target)

        # Observe 4 / 5 of the available features
        for ii in range(4):
            observation = fake_obs_test[ii]
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])

        if not focus_on_pose:
            # Since up to now we had identical evidence for both cube and house, we give
            # the house an edge now so we test for it and get the expected result.
            graph_lm.current_mlh["graph_id"] = "new_object1"
            graph_lm._hypotheses["new_object1"].evidence += 1

        # Based on most recent observation, propose the most misaligned graph
        # sub-regions
        graph_lm.gsg.focus_on_pose = focus_on_pose
        target_loc_id = graph_lm.gsg._compute_graph_mismatch()

        target_graph = graph_lm.get_graph(target_object)
        first_input_channel = graph_lm.get_input_channels_in_graph(target_object)[0]
        target_loc = target_graph[first_input_channel].pos[target_loc_id]

        assert np.all(np.isclose(target_loc, self.fake_obs_house[4].location)), (
            "Should propose testing 5th (indexed from 0) location on object"
        )

    def test_hypothesis_testing_proposal_for_id(self):
        """Test that the LM correctly predicts a location on a graph to test.

        Test that the LM correctly predicts a location on a graph to test, given
        the mismatch between the top two most likely graphs.

        The "square" object is made up of four simple features, while the "house"
        object is a 2D square with an additional, 5th feature above (like the point
        on a house's roof); these two objects are a 2D analogue for distinguishing e.g.
        a mug with a handle from a cylindrical can.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_house)

        graph_lm = self.get_elm_with_two_fake_objects(
            self.fake_obs_square,
            self.fake_obs_house,
            initial_possible_poses=[[0, 0, 0]],  # Note we isolate the influence of
            # ambiguous pose on the hypothesis testing
            gsg=EvidenceGoalGenerator(**self.default_gsg_config),
        )

        self._evaluate_target_location(
            graph_lm, fake_obs_test, target_object="new_object1"
        )

    def test_hypothesis_testing_proposal_for_id_with_transformation(self):
        """Test that the LM correctly predicts a location on a graph to test.

        Test that the LM correctly predicts a location on a graph to test, given
        the mismatch between the top two most likely graphs.

        As for test_hypothesis_testing_proposal_for_id, but in this case, the
        "house" object has been rotated and translated in the environment, and we check
        that the policy still proposes the correct location in object-centric
        coordinates.
        """
        fake_obs_test = copy.deepcopy(self.fake_percept_house_trans)

        graph_lm = self.get_elm_with_two_fake_objects(
            self.fake_obs_square,
            self.fake_obs_house,
            initial_possible_poses=[[45, 75, 190]],  # Note we isolate the influence of
            # ambiguous pose on the hypothesis testing
            gsg=EvidenceGoalGenerator(**self.default_gsg_config),
        )

        self._evaluate_target_location(
            graph_lm, fake_obs_test, target_object="new_object1"
        )

    def test_hypothesis_testing_proposal_for_pose(self):
        """Test that the LM correctly predicts a location on a graph to test.

        Test that the LM correctly predicts a location on a graph to test, given
        the mismatch between the top two most likely poses of a given object.

        In this case, the house can be either right-side up or upside-down, so the
        policy should select the tip of the house's roof to disambiguate this.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_house)

        # Only trained on one object
        graph_lm = self.get_elm_with_fake_object(
            self.fake_obs_house,
            initial_possible_poses=[[0, 0, 0], [0, 0, 180]],
            # Note pose *is* ambiguous in this unti test, vs. in proposal_for_id; in
            # particular, house can either be right-side up, or upside-down (rotated
            # about z)
            gsg=EvidenceGoalGenerator(**self.default_gsg_config),
        )

        self._evaluate_target_location(
            graph_lm, fake_obs_test, target_object="new_object0", focus_on_pose=True
        )

    def test_hypothesis_testing_proposal_for_pose_with_transformation(self):
        """Test that the LM correctly predicts a location on a graph to test.

        Test that the LM correctly predicts a location on a graph to test, given
        the mismatch between the top two most likely poses of a given object.

        In this case, the house can be either right-side up or upside-down, so the
        policy should select the tip of the house's roof to disambiguate this.
        In addition, the house has been transformed in environmental coordinates
        (translated and rotated), and we confirm that the object-centric target
        point is still correct.
        """
        fake_obs_test = copy.deepcopy(self.fake_percept_house_trans)

        # Only trained on one object
        graph_lm = self.get_elm_with_fake_object(
            self.fake_obs_house,
            initial_possible_poses=[[45, 75, 190], [315, 285, 10]],
            # Note pose *is* ambiguous in this unti test, vs. in proposal_for_id; in
            # particular, house can either be right-side up, or upside-down (was rotated
            # about z before the additional complex transformation was applied)
            gsg=EvidenceGoalGenerator(**self.default_gsg_config),
        )

        self._evaluate_target_location(
            graph_lm, fake_obs_test, target_object="new_object0", focus_on_pose=True
        )

    def test_different_features_still_recognized(self):
        """Test that the object is still recognized if features don't match.

        The evidence LM just uses features as an aid to recognize objects faster
        but should still be able to recognize an object if features don't match
        as long as the morphology matches.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].non_morphological_features["hsv"] = [0.5, 1, 1]
        fake_obs_test[1].non_morphological_features["hsv"] = [0.9, 1, 1]
        fake_obs_test[2].non_morphological_features["hsv"] = [0.5, 1, 1]
        fake_obs_test[3].non_morphological_features["hsv"] = [0.7, 1, 1]

        for i in range(4):
            fake_obs_test[i].non_morphological_features["principal_curvatures_log"] = [
                10,
                10,
            ]

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.reset_stm()
        graph_lm.fixme_reset_ground_truth(primary_target=self.placeholder_target)
        # We start at evidence 0 since we don't get feature evidence at initialization
        for target_evidence, observation in enumerate(fake_obs_test):
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])
            self.assertEqual(
                len(graph_lm.get_possible_matches()),
                1,
                "Should have exactly 1 possible match.",
            )
            self.assertEqual(
                graph_lm.get_current_mlh()["evidence"],
                target_evidence,
                "Since none of the features match we should only be getting evidence "
                "from morphology which is 1 at each step (since m matches perfectly).",
            )

        self.assertListEqual(
            list(graph_lm.get_current_mlh()["rotation"].as_euler("xyz", degrees=True)),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    def test_different_pose_features_not_recognized_elm(self):
        """Test that the object is not recognized if pose features don't match."""
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[0].morphological_features["pose_vectors"] = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.reset_stm()
        graph_lm.fixme_reset_ground_truth(primary_target=self.placeholder_target)
        for step, observation in enumerate(fake_obs_test):
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])
            if step == 0:
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    1,
                    "Should have one possible match at first step.",
                )
                self.assertEqual(
                    graph_lm.get_current_mlh()["evidence"],
                    1,
                    "First step should have evidence 1 since features match and poses"
                    " are just being initialized.",
                )
            elif step == 1:
                self.assertGreater(
                    graph_lm.get_current_mlh()["evidence"],
                    1,
                    "Second step should still have evidence >1 since initialized pose"
                    " is still possible.",
                )
            elif step == 2:
                self.assertLess(
                    graph_lm.get_current_mlh()["evidence"],
                    1,
                    "Third step should have evidence <1 since initialized pose"
                    " is not valid anymore -> subtracting 1.",
                )
            elif step == 3:
                self.assertLess(
                    graph_lm.get_current_mlh()["evidence"],
                    0,
                    "Fourth step should have evidence <0 since initialized pose"
                    " is not valid anymore -> subtracting 1.",
                )
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    0,
                    "Should have no possible matches.",
                )

    def test_moving_off_object_and_back_elm(self):
        """Test that the object is still recognized after moving off the object.

        Test that the object is still recognized after moving off the object
        and that evidence is not incremented in that step.

        TODO: since the monty class checks use_state in combine_inputs it doesn't make
        much sense to test this here anymore with an isolated LM.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        fake_obs_test[1].location = [1, 2, 1]
        fake_obs_test[1].morphological_features["on_object"] = 0
        fake_obs_test[1].use_state = False

        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.reset_stm()
        graph_lm.fixme_reset_ground_truth(primary_target=self.placeholder_target)
        target_evidence = 1
        for step, observation in enumerate(fake_obs_test):
            if not observation.use_state:
                pass
            else:
                graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
                graph_lm.matching_step(self.ctx, [observation])
                # If we are off the object, no evidence should be added
                if observation.morphological_features["on_object"] != 0 and step > 0:
                    target_evidence += 2
                self.assertEqual(
                    len(graph_lm.get_possible_matches()),
                    1,
                    "Should have exactly one possible match.",
                )
                self.assertEqual(
                    graph_lm.get_possible_matches()[0],
                    "new_object0",
                    "Should match to new_object0.",
                )
                self.assertEqual(
                    graph_lm.get_current_mlh()["graph_id"],
                    "new_object0",
                    "new_object0 should be the mlh at every step.",
                )
                self.assertEqual(
                    graph_lm.get_current_mlh()["evidence"],
                    target_evidence,
                    "Since we have exact matches, evidence should be incremented by "
                    "2 at each step (1 for pose match and 1 for feature match).",
                )

        self.assertListEqual(
            list(graph_lm.get_current_mlh()["rotation"].as_euler("xyz", degrees=True)),
            [0, 0, 0],
            "Should recognize rotation 0, 0, 0.",
        )

    # =================== Model splitting ===================

    def test_positive_evidence_cluster_detected(self):
        """A tight positive cluster separate from the rest is detected."""
        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)
        rng = np.random.RandomState(42)
        # The well-matching points form a tight positive-evidence cluster;
        # the remaining points are spread out over negative values (they do
        # not form a mode of their own).
        positive_cluster = rng.normal(2.0, 0.1, 50)
        rest = rng.uniform(-2.0, 0.0, 50)

        self.assertTrue(
            graph_lm._detect_positive_evidence_cluster(
                np.concatenate([positive_cluster, rest])
            ),
            "A tight positive-evidence cluster that is well separated from "
            "the remaining values should be detected.",
        )

    def test_unclustered_evidence_distribution_not_split(self):
        """Distributions without a distinct positive cluster are not split."""
        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)
        rng = np.random.RandomState(42)

        self.assertFalse(
            graph_lm._detect_positive_evidence_cluster(rng.normal(0.0, 1.0, 200)),
            "A Gaussian distribution centered at 0 should not be detected "
            "as containing a positive-evidence cluster.",
        )
        self.assertFalse(
            graph_lm._detect_positive_evidence_cluster(rng.uniform(-1.0, 1.0, 200)),
            "A uniform distribution should not be detected as containing a "
            "positive-evidence cluster.",
        )
        self.assertFalse(
            graph_lm._detect_positive_evidence_cluster(np.ones(10)),
            "All-positive values leave no second component to split off.",
        )
        self.assertFalse(
            graph_lm._detect_positive_evidence_cluster(np.full(10, -1.0)),
            "All-negative values contain no positive-evidence cluster.",
        )

    def test_tiny_cluster_does_not_trigger_split(self):
        """A few positive outliers should not count as a cluster."""
        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)
        rng = np.random.RandomState(42)
        values = np.concatenate([rng.normal(-1.0, 0.05, 97), np.full(3, 5.0)])

        self.assertFalse(
            graph_lm._detect_positive_evidence_cluster(values),
            "A positive cluster holding less than split_min_cluster_fraction "
            "of the values should not trigger a split.",
        )

    def test_spatially_isolated_points_dropped_from_split_component(self):
        """Candidate points without a nearby candidate neighbor are dropped."""
        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)
        graph_lm.split_spatial_neighbor_distance = 0.003
        locations = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.002, 0.0, 0.0],
                [0.004, 0.0, 0.0],
                [0.1, 0.0, 0.0],  # isolated: nearest candidate is ~0.096 away
            ]
        )
        candidate_ids = np.arange(4)

        kept = graph_lm._drop_spatially_isolated_points(candidate_ids, locations)

        self.assertListEqual(
            list(kept),
            [0, 1, 2],
            "Only points within split_spatial_neighbor_distance of another "
            "candidate should be kept.",
        )
        self.assertEqual(
            graph_lm._drop_spatially_isolated_points(
                np.array([3]), locations
            ).size,
            0,
            "A single candidate point has no neighbors and should be dropped.",
        )

    def test_split_graph_partitions_model(self):
        """_split_graph divides a graph's nodes into two component graphs."""
        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)
        memory = graph_lm.graph_memory
        model = memory.get_graph("new_object0", "patch")
        pos = np.asarray(model.pos).copy()
        n_nodes = pos.shape[0]
        self.assertGreaterEqual(n_nodes, 4, "Should have learned all 4 points.")
        first_ids = np.array([0, 1])
        second_ids = np.arange(2, n_nodes)

        success = memory._split_graph(
            graph_id="new_object0",
            node_ids_per_channel={"patch": [first_ids, second_ids]},
            new_graph_ids=["new_object0_component_1", "new_object0_component_2"],
        )

        self.assertTrue(success, "The split should succeed.")
        self.assertNotIn(
            "new_object0",
            memory.get_memory_ids(),
            "The source graph should be removed from memory.",
        )
        for component_id, node_ids in [
            ("new_object0_component_1", first_ids),
            ("new_object0_component_2", second_ids),
        ]:
            self.assertIn(
                component_id,
                memory.get_memory_ids(),
                f"{component_id} should be registered in memory.",
            )
            component_model = memory.get_graph(component_id, "patch")
            component_pos = np.asarray(component_model.pos)
            self.assertEqual(
                component_pos.shape[0],
                len(node_ids),
                f"{component_id} should contain exactly its subset of nodes.",
            )
            for location in pos[node_ids]:
                self.assertTrue(
                    np.any(np.all(np.isclose(component_pos, location), axis=1)),
                    f"{component_id} should contain the source location "
                    f"{location} unchanged (same reference frame).",
                )
            self.assertEqual(
                component_model._match_evidence,
                {},
                "Component models should start with empty match-evidence "
                "metadata.",
            )

    def test_model_splits_after_full_annotation_with_positive_cluster(self):
        """A fully annotated model with a positive-evidence cluster is split.

        Runs matching steps until the persistence ("high confidence")
        condition is met, then overwrites the model's annotations so that
        part of the model forms a tight positive-evidence cluster and the
        rest holds non-positive evidence. The next matching step should
        detect this and split the model into two component graphs.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)
        graph_lm.required_symmetry_evidence = 3
        # Prevent a split while the persistence condition is being reached.
        graph_lm.split_cluster_separation_threshold = np.inf
        # The fake object's points are 1m apart, so widen the spatial
        # coherence radius to cover each component's nearest neighbors.
        graph_lm.split_spatial_neighbor_distance = 1.5

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.reset_stm()
        graph_lm.fixme_reset_ground_truth(primary_target=self.placeholder_target)

        for step in range(12):
            observation = fake_obs_test[step % 4]
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])
        self.assertListEqual(
            list(graph_lm._persistent_hypothesis_ids.keys()),
            ["new_object0"],
            "Persistent hypotheses should be narrowed down to new_object0.",
        )

        model = graph_lm.graph_memory.get_graph("new_object0", "patch")
        pos = np.asarray(model.pos).copy()
        # The nodes at (0,0,0) and (1,0,0) form the positive-evidence
        # cluster, the remaining nodes the poorly matching rest.
        high_ids = [
            int(np.argmin(np.linalg.norm(pos - location, axis=1)))
            for location in (np.zeros(3), np.array([1.0, 0.0, 0.0]))
        ]
        low_ids = [i for i in range(pos.shape[0]) if i not in high_ids]
        model._match_evidence.clear()
        model.annotate_match_evidence(high_ids, [3.0] * len(high_ids))
        model.annotate_match_evidence(low_ids, [-3.0] * len(low_ids))
        # Freeze the annotations so the next step's annotation of matched
        # nodes does not move them.
        model.match_evidence_smoothing = 0.0
        graph_lm.split_cluster_separation_threshold = 4.0

        graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
        graph_lm.matching_step(self.ctx, [fake_obs_test[0]])

        known_ids = graph_lm.get_all_known_object_ids()
        self.assertNotIn(
            "new_object0",
            known_ids,
            "The source graph should have been removed by the split.",
        )
        for component_id, node_ids in [
            ("new_object0_component_1", high_ids),
            ("new_object0_component_2", low_ids),
        ]:
            self.assertIn(
                component_id,
                known_ids,
                f"{component_id} should be in memory after the split.",
            )
            component_model = graph_lm.graph_memory.get_graph(
                component_id, "patch"
            )
            component_pos = np.asarray(component_model.pos)
            self.assertEqual(
                component_pos.shape[0],
                len(node_ids),
                f"{component_id} should contain its evidence mode's nodes.",
            )
            for location in pos[node_ids]:
                self.assertTrue(
                    np.any(np.all(np.isclose(component_pos, location), axis=1)),
                    f"{component_id} should contain the source location "
                    f"{location} unchanged (same reference frame).",
                )
            _, counts = component_model.get_match_evidence()
            self.assertTrue(
                np.all(counts == 0),
                "Component models should start unannotated.",
            )
            self.assertIn(
                component_id,
                graph_lm._hypotheses,
                "Components should inherit the source graph's hypotheses.",
            )
        self.assertEqual(
            graph_lm.symmetry_evidence,
            0,
            "Symmetry evidence should be reset by the split.",
        )
        self.assertEqual(
            graph_lm._persistent_hypothesis_ids,
            {},
            "Persistent hypotheses should be reset by the split.",
        )

    def test_split_skipped_when_no_spatially_coherent_component(self):
        """No split occurs when spatial filtering empties a component.

        Same setup as the successful split test, but with the spatial
        coherence radius left at a value smaller than the fake object's
        1m point spacing: every candidate point is spatially isolated, so
        both components come up empty and the model must stay intact.
        """
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)
        graph_lm.required_symmetry_evidence = 3
        # Prevent a split while the persistence condition is being reached.
        graph_lm.split_cluster_separation_threshold = np.inf
        # Smaller than the 1m spacing between the fake object's points, so
        # no candidate point has a nearby candidate neighbor.
        graph_lm.split_spatial_neighbor_distance = 0.5

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.reset_stm()
        graph_lm.fixme_reset_ground_truth(primary_target=self.placeholder_target)

        for step in range(12):
            observation = fake_obs_test[step % 4]
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [observation])

        model = graph_lm.graph_memory.get_graph("new_object0", "patch")
        pos = np.asarray(model.pos)
        high_ids = [
            int(np.argmin(np.linalg.norm(pos - location, axis=1)))
            for location in (np.zeros(3), np.array([1.0, 0.0, 0.0]))
        ]
        low_ids = [i for i in range(pos.shape[0]) if i not in high_ids]
        model._match_evidence.clear()
        model.annotate_match_evidence(high_ids, [3.0] * len(high_ids))
        model.annotate_match_evidence(low_ids, [-3.0] * len(low_ids))
        model.match_evidence_smoothing = 0.0
        graph_lm.split_cluster_separation_threshold = 4.0

        graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
        graph_lm.matching_step(self.ctx, [fake_obs_test[0]])

        self.assertIn(
            "new_object0",
            graph_lm.get_all_known_object_ids(),
            "The model should not be split when its candidate points are "
            "spatially isolated.",
        )

    def test_stats_collection_after_split_does_not_raise(self):
        """Collecting stats works when the previous MLH's graph was split away.

        When a graph is split, its hypotheses-updater telemetry is removed,
        but `previous_mlh` may still reference the old graph id when stats
        are collected later in the same matching step.
        """
        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn)
        graph_lm.previous_mlh = dict(graph_lm.current_mlh)
        graph_lm.previous_mlh["graph_id"] = "spoon"
        self.assertNotIn("spoon", graph_lm.hypotheses_updater_telemetry)

        graph_lm.collect_stats_to_save()  # Should not raise.

        self.assertEqual(
            graph_lm.buffer.stats.get("mlh_prediction_error", []),
            [],
            "No prediction error should be logged for a split-away graph.",
        )

    # =================== Dense exploration policy ===================

    def get_elm_after_dense_exploration_steps(self, num_steps=2):
        """Train an LM with a dense-exploration GSG and run matching steps.

        Returns:
            The learning module (with `graph_lm.gsg` set to the
            DenseExplorationGoalGenerator).
        """
        gsg = DenseExplorationGoalGenerator(**self.default_gsg_config)
        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn, gsg=gsg)

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.reset_stm()
        graph_lm.fixme_reset_ground_truth(primary_target=self.placeholder_target)
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        for step in range(num_steps):
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [fake_obs_test[step % 4]])
        return graph_lm

    def get_persistent_elm_after_dense_exploration_steps(self):
        """Like get_elm_after_dense_exploration_steps, but reach persistence.

        Failed-jump annotations (like the annotation of matched nodes) are
        gated on the LM's persistent hypotheses being narrowed down to the
        target graph, so tests of those annotations need an LM that has
        reached this "high confidence" state. Runs matching steps until the
        persistence condition is met and then clears the annotations that
        accumulated along the way, so tests start from a clean, deterministic
        annotation state.

        Returns:
            The learning module (with `graph_lm.gsg` set to the
            DenseExplorationGoalGenerator) whose persistent hypotheses are
            narrowed down to "new_object0".
        """
        gsg = DenseExplorationGoalGenerator(**self.default_gsg_config)
        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn, gsg=gsg)
        graph_lm.required_symmetry_evidence = 3
        # Prevent a split while the persistence condition is being reached.
        graph_lm.split_cluster_separation_threshold = np.inf

        graph_lm.mode = ExperimentMode.EVAL
        graph_lm.reset_stm()
        graph_lm.fixme_reset_ground_truth(primary_target=self.placeholder_target)
        fake_obs_test = copy.deepcopy(self.fake_obs_learn)
        for step in range(12):
            graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            graph_lm.matching_step(self.ctx, [fake_obs_test[step % 4]])
        self.assertListEqual(
            list(graph_lm._persistent_hypothesis_ids.keys()),
            ["new_object0"],
            "Persistent hypotheses should be narrowed down to new_object0.",
        )

        model = graph_lm.graph_memory.get_graph("new_object0", "patch")
        model._match_evidence.clear()
        gsg._goals_pending_jump_outcome = []
        return graph_lm

    def test_dense_exploration_targets_nearest_unannotated_point(self):
        """The dense-exploration policy targets the sole unannotated point."""
        graph_lm = self.get_elm_after_dense_exploration_steps()
        gsg = graph_lm.gsg

        model = graph_lm.graph_memory.get_graph("new_object0", "patch")
        pos = np.asarray(model.pos)
        # Annotate every node except the one at (1,1,1).
        target_node = int(
            np.argmin(np.linalg.norm(pos - np.array([1.0, 1.0, 1.0]), axis=1))
        )
        annotated = [i for i in range(pos.shape[0]) if i != target_node]
        model.annotate_match_evidence(annotated, [1.0] * len(annotated))

        self.assertEqual(
            gsg._get_unannotated_node(),
            target_node,
            "The only unannotated node should be selected as the target.",
        )

        goal = gsg._generate_goal([copy.deepcopy(self.fake_obs_learn[1])])
        self.assertIsNotNone(goal, "A Goal should be generated.")
        self.assertTrue(
            np.allclose(goal.info["proposed_surface_loc"], pos[target_node]),
            "The Goal should target the unannotated model point (identity "
            "pose, so model and body coordinates coincide).",
        )

    def test_dense_exploration_no_goal_when_fully_annotated(self):
        """No Goal is produced once every model point has been annotated."""
        graph_lm = self.get_elm_after_dense_exploration_steps()
        gsg = graph_lm.gsg

        model = graph_lm.graph_memory.get_graph("new_object0", "patch")
        n_nodes = np.asarray(model.pos).shape[0]
        model.annotate_match_evidence(list(range(n_nodes)), [1.0] * n_nodes)

        self.assertIsNone(
            gsg._get_unannotated_node(),
            "No target should be proposed when all points are annotated.",
        )
        self.assertIsNone(
            gsg._generate_goal([copy.deepcopy(self.fake_obs_learn[1])]),
            "The None Goal should be generated when all points are annotated.",
        )

    def test_dense_exploration_annotates_failed_jump_target(self):
        """A model point whose jump was undone receives negative evidence."""
        graph_lm = self.get_persistent_elm_after_dense_exploration_steps()
        gsg = graph_lm.gsg

        goal = gsg._generate_goal([copy.deepcopy(self.fake_obs_learn[1])])
        self.assertIsNotNone(goal, "A Goal should be generated.")
        self.assertIn(
            goal,
            gsg._goals_pending_jump_outcome,
            "The emitted Goal should await its jump outcome.",
        )
        target_node = goal.info["target_loc_id"]
        model = graph_lm.graph_memory.get_graph(
            goal.info["target_graph_id"], goal.info["target_input_channel"]
        )
        _, counts_before = model.get_match_evidence(node_ids=[target_node])
        self.assertEqual(
            counts_before[0],
            0,
            "The targeted point should not be annotated yet.",
        )

        # Simulate the motor system undoing the jump to the Goal (i.e. the
        # targeted point does not exist in the world).
        goal.info["jump_failed"] = True
        gsg._annotate_failed_jump_targets()

        evidence, counts = model.get_match_evidence(node_ids=[target_node])
        self.assertEqual(
            counts[0],
            1,
            "The failed jump's target should now count as annotated.",
        )
        self.assertEqual(
            evidence[0],
            -1.0,
            "The failed jump's target should have received -1 evidence.",
        )
        self.assertNotIn(
            goal,
            gsg._goals_pending_jump_outcome,
            "The resolved Goal should no longer be pending.",
        )

    def test_dense_exploration_ignores_successful_jump_target(self):
        """A model point whose jump succeeded receives no failure annotation."""
        graph_lm = self.get_persistent_elm_after_dense_exploration_steps()
        gsg = graph_lm.gsg

        goal = gsg._generate_goal([copy.deepcopy(self.fake_obs_learn[1])])
        self.assertIsNotNone(goal, "A Goal should be generated.")
        target_node = goal.info["target_loc_id"]
        model = graph_lm.graph_memory.get_graph(
            goal.info["target_graph_id"], goal.info["target_input_channel"]
        )

        goal.info["jump_failed"] = False
        gsg._annotate_failed_jump_targets()

        _, counts = model.get_match_evidence(node_ids=[target_node])
        self.assertEqual(
            counts[0],
            0,
            "A successful jump should not annotate the targeted point.",
        )
        self.assertNotIn(
            goal,
            gsg._goals_pending_jump_outcome,
            "The resolved Goal should no longer be pending.",
        )

    def test_failed_jump_before_confidence_is_discarded(self):
        """No -1 annotation while the LM has not reached high confidence.

        The accumulation of match evidence (_annotate_matched_nodes) only
        starts once the LM's persistent hypotheses are narrowed down to one
        object. Failed-jump evidence must respect the same gating: a Goal
        computed from a not-yet-confident hypothesis is not reliable evidence
        against the targeted model points, so its failure is discarded
        without annotating anything.
        """
        # Two matching steps: an MLH exists but persistence is not reached.
        graph_lm = self.get_elm_after_dense_exploration_steps()
        gsg = graph_lm.gsg
        self.assertEqual(
            len(graph_lm._persistent_hypothesis_ids),
            0,
            "The LM should not have reached the persistence condition yet.",
        )

        goal = gsg._generate_goal([copy.deepcopy(self.fake_obs_learn[1])])
        self.assertIsNotNone(goal, "A Goal should be generated.")
        model = graph_lm.graph_memory.get_graph(
            goal.info["target_graph_id"], goal.info["target_input_channel"]
        )

        goal.info["jump_failed"] = True
        gsg._annotate_failed_jump_targets()

        _, counts = model.get_match_evidence()
        self.assertTrue(
            np.all(counts == 0),
            "No point should be annotated before the LM is confident.",
        )
        self.assertNotIn(
            goal,
            gsg._goals_pending_jump_outcome,
            "The discarded Goal should no longer be pending.",
        )

    def test_failed_jump_annotates_local_region(self):
        """All points within the location tolerance of the target receive -1.

        A failed jump is evidence against the local region of the model, not
        just the single targeted node: had any point within the Goal's
        location tolerance existed in the world, the jump would have found
        the object.
        """
        graph_lm = self.get_persistent_elm_after_dense_exploration_steps()
        gsg = graph_lm.gsg
        # Widen the tolerance so it covers the nodes at (1,1,1) and (1,1,0)
        # (distance 1.0) but not (1,0,0) (distance ~1.41) or (0,0,0).
        gsg.goal_tolerances["location"] = 1.2

        goal = gsg._generate_goal([copy.deepcopy(self.fake_obs_learn[1])])
        self.assertIsNotNone(goal, "A Goal should be generated.")
        model = graph_lm.graph_memory.get_graph(
            goal.info["target_graph_id"], goal.info["target_input_channel"]
        )
        pos = np.asarray(model.pos)
        target_node = int(
            np.argmin(np.linalg.norm(pos - np.array([1.0, 1.0, 1.0]), axis=1))
        )
        neighbor_node = int(
            np.argmin(np.linalg.norm(pos - np.array([1.0, 1.0, 0.0]), axis=1))
        )
        goal.info["target_loc_id"] = target_node

        goal.info["jump_failed"] = True
        gsg._annotate_failed_jump_targets()

        evidence, counts = model.get_match_evidence()
        for node in (target_node, neighbor_node):
            self.assertEqual(
                counts[node],
                1,
                "Points within the tolerance of the target should have been "
                "annotated.",
            )
            self.assertEqual(
                evidence[node],
                -1.0,
                "Points within the tolerance of the target should have "
                "received -1 evidence.",
            )
        outside = [
            i for i in range(pos.shape[0]) if i not in (target_node, neighbor_node)
        ]
        self.assertTrue(
            np.all(counts[outside] == 0),
            "Points outside the tolerance of the target should not have been "
            "annotated.",
        )

    def test_dense_exploration_full_loop_annotates_failed_jump(self):
        """The -1 annotation works end-to-end through a real motor policy.

        A Goal emitted by the GSG is enacted by a real JumpToGoal policy: the
        agent is teleported to the Goal's location, nothing is visible there
        (a failed jump), the policy undoes the jump and records the outcome
        on the Goal, and the next matching step folds -1 into the targeted
        model point's annotation.
        """
        graph_lm = self.get_persistent_elm_after_dense_exploration_steps()
        gsg = graph_lm.gsg

        model = graph_lm.graph_memory.get_graph("new_object0", "patch")
        pos = np.asarray(model.pos)
        # Leave only the node at (1,1,1) unannotated so the Goal
        # deterministically targets it (and the final matching step, whose
        # observation is at (0,0,0), cannot touch it with its own annotation
        # of matched nodes).
        target_node = int(
            np.argmin(np.linalg.norm(pos - np.array([1.0, 1.0, 1.0]), axis=1))
        )
        annotated = [i for i in range(pos.shape[0]) if i != target_node]
        model.annotate_match_evidence(annotated, [1.0] * len(annotated))

        goal = gsg._generate_goal([copy.deepcopy(self.fake_obs_learn[1])])
        self.assertIsNotNone(goal, "A Goal should be generated.")
        self.assertEqual(goal.info["target_loc_id"], target_node)

        agent_id = AgentID("agent_id_0")

        def state_at(position):
            return MotorSystemState(
                {
                    agent_id: AgentState(
                        sensors={
                            SensorID("sensor_id_0"): SensorState(
                                position=(0.0, 0.0, 0.0), rotation=qt.one
                            )
                        },
                        position=position,
                        rotation=qt.one,
                    )
                }
            )

        policy = JumpToGoal(agent_id, SensorID("view_finder"))
        result = policy(
            ctx=Mock(),
            observations=Mock(),
            state=state_at((0.0, 0.0, 0.0)),
            percept=Mock(),
            goal=goal,
        )
        self.assertGreater(
            len(result.actions), 0, "The policy should enact the jump."
        )
        with patch(
            "tbp.monty.frameworks.models.motor_policies."
            "PositioningProcedure.depth_at_center",
            return_value=1.0,  # Nothing visible at the target: a failed jump.
        ):
            result = policy(
                ctx=Mock(),
                observations=Mock(),
                # The jump was executed: the agent is at the Goal's location.
                state=state_at(tuple(goal.location)),
                percept=Mock(),
                goal=None,
            )
        self.assertGreater(
            len(result.actions), 0, "The policy should undo the jump."
        )
        self.assertIs(
            goal.info["jump_failed"],
            True,
            "The policy should record the failure on the Goal.",
        )

        # The next matching step (which steps the GSG) applies the annotation.
        graph_lm.add_lm_processing_to_buffer_stats(lm_processed=True)
        graph_lm.matching_step(self.ctx, [copy.deepcopy(self.fake_obs_learn[0])])

        evidence, counts = model.get_match_evidence(node_ids=[target_node])
        self.assertEqual(
            counts[0],
            1,
            "The failed jump's target should have been annotated once.",
        )
        self.assertEqual(
            evidence[0],
            -1.0,
            "The failed jump's target should have received -1 evidence.",
        )

    def test_failed_jump_folds_into_existing_annotation(self):
        """The -1 is folded into a previously annotated point via the EMA."""
        graph_lm = self.get_persistent_elm_after_dense_exploration_steps()
        gsg = graph_lm.gsg

        goal = gsg._generate_goal([copy.deepcopy(self.fake_obs_learn[1])])
        self.assertIsNotNone(goal, "A Goal should be generated.")
        target_node = goal.info["target_loc_id"]
        model = graph_lm.graph_memory.get_graph(
            goal.info["target_graph_id"], goal.info["target_input_channel"]
        )
        # The point was previously annotated with positive evidence (e.g. by
        # _annotate_matched_nodes) before the jump to it failed.
        model.annotate_match_evidence([target_node], [1.0])

        goal.info["jump_failed"] = True
        gsg._annotate_failed_jump_targets()

        evidence, counts = model.get_match_evidence(node_ids=[target_node])
        expected = 1.0 + model.match_evidence_smoothing * (
            gsg.failed_jump_evidence - 1.0
        )
        self.assertEqual(
            counts[0],
            2,
            "The failure should count as a second annotation of the point.",
        )
        self.assertAlmostEqual(
            evidence[0],
            expected,
            msg="The -1 should be folded into the existing annotation with "
            "the model's EMA smoothing.",
        )

    def test_repeated_failed_jumps_drive_annotation_towards_minus_one(self):
        """Repeated failures accumulate, pulling the stored evidence to -1."""
        graph_lm = self.get_persistent_elm_after_dense_exploration_steps()
        gsg = graph_lm.gsg

        goal = gsg._generate_goal([copy.deepcopy(self.fake_obs_learn[1])])
        self.assertIsNotNone(goal, "A Goal should be generated.")
        target_node = goal.info["target_loc_id"]
        model = graph_lm.graph_memory.get_graph(
            goal.info["target_graph_id"], goal.info["target_input_channel"]
        )
        # Start from a strongly positive annotation.
        model.annotate_match_evidence([target_node], [1.0])

        # Re-enact the (already emitted) Goal once per failure below.
        gsg._goals_pending_jump_outcome = []

        num_failures = 30
        previous = 1.0
        for _ in range(num_failures):
            gsg._goals_pending_jump_outcome.append(goal)
            goal.info["jump_failed"] = True
            gsg._annotate_failed_jump_targets()
            evidence, _ = model.get_match_evidence(node_ids=[target_node])
            self.assertLess(
                evidence[0],
                previous,
                "Each failure should decrease the stored evidence.",
            )
            previous = evidence[0]

        evidence, counts = model.get_match_evidence(node_ids=[target_node])
        self.assertEqual(
            counts[0],
            num_failures + 1,
            "Each failure should count as an annotation of the point.",
        )
        self.assertLess(
            evidence[0],
            -0.9,
            "Repeated failures should pull the evidence close to -1.",
        )

    def test_failed_jump_annotation_survives_newer_pending_goals(self):
        """A failed jump is annotated even after newer Goals were emitted.

        Reproduces the real timeline: the GSG steps once more (emitting a new
        Goal) before the motor system records the outcome of the previous
        jump. The older Goal must remain tracked until its outcome arrives.
        """
        graph_lm = self.get_persistent_elm_after_dense_exploration_steps()
        gsg = graph_lm.gsg
        observations = [copy.deepcopy(self.fake_obs_learn[1])]

        model = graph_lm.graph_memory.get_graph("new_object0", "patch")
        pos = np.asarray(model.pos)
        # Leave a single unannotated node so successive Goals
        # deterministically target the same point.
        target_node = int(
            np.argmin(np.linalg.norm(pos - np.array([1.0, 1.0, 1.0]), axis=1))
        )
        annotated = [i for i in range(pos.shape[0]) if i != target_node]
        model.annotate_match_evidence(annotated, [1.0] * len(annotated))

        first_goal = gsg._generate_goal(observations)
        self.assertEqual(first_goal.info["target_loc_id"], target_node)
        # The GSG steps again before the jump outcome is known: the pending
        # Goal is retained and a newer Goal is emitted.
        gsg._annotate_failed_jump_targets()
        second_goal = gsg._generate_goal(observations)
        self.assertEqual(second_goal.info["target_loc_id"], target_node)
        self.assertIn(
            first_goal,
            gsg._goals_pending_jump_outcome,
            "The unresolved Goal should still be tracked.",
        )

        # The motor system now records the failure of the first jump (the
        # second Goal was superseded before being enacted).
        first_goal.info["jump_failed"] = True
        gsg._annotate_failed_jump_targets()

        evidence, counts = model.get_match_evidence(node_ids=[target_node])
        self.assertEqual(
            counts[0],
            1,
            "The failed jump's target should have been annotated once.",
        )
        self.assertEqual(
            evidence[0],
            -1.0,
            "The failed jump's target should have received -1 evidence.",
        )
        self.assertNotIn(
            first_goal,
            gsg._goals_pending_jump_outcome,
            "The resolved Goal should no longer be pending.",
        )

    def test_failed_jump_target_is_not_retargeted(self):
        """A point whose jump failed counts as visited and is not re-proposed."""
        graph_lm = self.get_persistent_elm_after_dense_exploration_steps()
        gsg = graph_lm.gsg
        observations = [copy.deepcopy(self.fake_obs_learn[1])]

        model = graph_lm.graph_memory.get_graph("new_object0", "patch")
        n_nodes = np.asarray(model.pos).shape[0]
        # Leave a single unannotated node so the Goal deterministically
        # targets it.
        annotated = list(range(1, n_nodes))
        model.annotate_match_evidence(annotated, [1.0] * len(annotated))

        goal = gsg._generate_goal(observations)
        self.assertEqual(goal.info["target_loc_id"], 0)

        goal.info["jump_failed"] = True
        gsg._annotate_failed_jump_targets()

        self.assertIsNone(
            gsg._get_unannotated_node(),
            "The failed jump's target should now count as visited.",
        )
        self.assertIsNone(
            gsg._generate_goal(observations),
            "No Goal should target the failed jump's point again.",
        )

    def test_failed_jump_to_removed_graph_is_ignored(self):
        """A failure whose target graph no longer exists is safely skipped."""
        graph_lm = self.get_persistent_elm_after_dense_exploration_steps()
        gsg = graph_lm.gsg

        goal = gsg._generate_goal([copy.deepcopy(self.fake_obs_learn[1])])
        self.assertIsNotNone(goal, "A Goal should be generated.")
        # Simulate the graph being removed (e.g. by a split) before the jump
        # outcome arrived.
        goal.info["target_graph_id"] = "removed_object"
        goal.info["jump_failed"] = True

        gsg._annotate_failed_jump_targets()  # Should not raise.

        self.assertNotIn(
            goal,
            gsg._goals_pending_jump_outcome,
            "The resolved Goal should no longer be pending.",
        )

    def test_dense_exploration_can_generate_goal_every_step(self):
        """The dense-exploration policy never waits to generate a new Goal."""
        gsg = DenseExplorationGoalGenerator(**self.default_gsg_config)
        graph_lm = self.get_elm_with_fake_object(self.fake_obs_learn, gsg=gsg)

        self.assertTrue(
            gsg._check_need_new_output_goal(self.ctx, output_goal_achieved=False),
            "The policy should be able to emit a Goal on every step.",
        )
        self.assertTrue(
            gsg._check_need_new_output_goal(self.ctx, output_goal_achieved=True),
            "Achieving the previous Goal should not stop new Goals.",
        )

        # Before matching starts there is no meaningful MLH, so no target.
        graph_lm.reset_stm()
        self.assertIsNone(
            gsg._get_unannotated_node(),
            "No target should be proposed before matching has started.",
        )
