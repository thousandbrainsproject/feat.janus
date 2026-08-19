# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Create a pretrained model with deliberately mismerged object graphs.

Loads an existing pretrained Monty model and, for each object in graph memory,
superimposes a second copy of the object's points that has been rotated
relative to the original (as if two instances of the object had been
incorrectly merged into a single model during learning). The rotated copy's
pose-dependent features (surface normals and curvature directions) are rotated
along with the locations, mirroring what update_model does when it merges new
observations into an existing graph via apply_rf_transform_to_points.

The resulting model is saved as a new pretrained model directory, and can be
used to test forking/splitting of incorrectly merged models.
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.utils.spatial_arithmetics import (
    rotate_multiple_pose_dependent_features,
)

DEFAULT_SOURCE = (
    "~/tbp/results/monty/pretrained_models/pretrained_janus/banana_can/pretrained"
)
DEFAULT_OUTPUT = (
    "~/tbp/results/monty/pretrained_models/pretrained_janus/"
    "mismerged_banana_can/pretrained"
)
DEFAULT_MISMERGE_ROTATION = [15.0, 15.0, 15.0]  # xyz euler angles in degrees


def extract_features_dict(model) -> dict:
    """Extract features from a graph's feature matrix into a dict.

    Returns per-feature numpy arrays keyed by feature name, in the order they
    appear in the graph's feature_mapping. node_ids are skipped since they are
    regenerated when the graph is rebuilt.
    """
    x = np.array(model.x)
    features = {}
    for feature, (start, stop) in model.feature_mapping.items():
        if feature == "node_ids":
            continue
        features[feature] = x[:, start:stop]
    return features


def mismerge_object_model(model, rotation: Rotation) -> None:
    """Superimpose a rotated copy of the model's points onto itself, in place.

    The copy is rotated about the centroid of the model's points so both
    "instances" occupy the same region of space, and its pose-dependent
    features (surface normals + curvature directions) are rotated to match.
    The adjacency graph is then rebuilt with the model's original k_n and
    graph_delta_thresholds, so points of the rotated copy that are
    indistinguishable from existing points (in location and feature space) are
    dropped, just as they would be during (mis)merged learning.
    """
    original_locations = np.array(model.pos, dtype=np.float64)
    original_features = extract_features_dict(model)
    num_original = original_locations.shape[0]

    centroid = original_locations.mean(axis=0)
    rotated_locations = rotation.apply(original_locations - centroid) + centroid
    rotated_features = rotate_multiple_pose_dependent_features(
        copy.deepcopy(original_features), rotation
    )

    all_locations = np.vstack([original_locations, rotated_locations])
    all_features = {
        feature: np.vstack([original_features[feature], rotated_features[feature]])
        for feature in original_features
    }

    new_graph = model._build_adjacency_graph(
        all_locations,
        all_features,
        k_n=model.k_n,
        graph_delta_thresholds=model.graph_delta_thresholds,
        old_graph_index=num_original,
    )
    model.set_graph(new_graph)
    if model.has_ppf:
        model.add_ppf_to_graph()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_SOURCE,
        help="Directory containing the source pretrained model.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Directory to save the mismerged model.pt to",
    )
    parser.add_argument(
        "--rotation",
        type=float,
        nargs=3,
        default=DEFAULT_MISMERGE_ROTATION,
        help="xyz euler rotation (degrees) of the superimposed copy",
    )
    args = parser.parse_args()

    source_path = Path(args.source).expanduser() / "model.pt"
    output_dir = Path(args.output).expanduser()
    output_path = output_dir / "model.pt"

    rotation = Rotation.from_euler("xyz", args.rotation, degrees=True)

    print(f"Loading pretrained model from {source_path}")
    state_dict = torch.load(source_path, weights_only=False)

    for lm_id, lm_dict in state_dict["lm_dict"].items():
        for object_id, channels in lm_dict["graph_memory"].items():
            for channel_id, model in channels.items():
                num_before = model.num_nodes
                mismerge_object_model(model, rotation)
                print(
                    f"LM {lm_id} / {object_id} / {channel_id}: "
                    f"{num_before} -> {model.num_nodes} points "
                    f"(rotated copy superimposed at "
                    f"{args.rotation} degrees xyz)"
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_path)
    print(f"Saved mismerged model to {output_path}")


if __name__ == "__main__":
    main()
