# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Create rotated variants of pretrained Monty object models.

For every object model stored in a pretrained `model.pt`, this script
creates `--num_rotations` variants (e.g. `banana_1_rot2`), each rotated by
an independent uniformly random rotation about the object's centroid.
Pose-dependent features are rotated along with the point positions:
surface normals, the three pose vectors per node, and the displacement
edge attributes. Pose-independent features (hsv, curvatures, depth, ...)
are left untouched. The output model directory contains only the variant
objects; the id mappings are rewritten so a detected variant still counts
as the original target object.

Example:
    python scripts/rotate_pretrained_models.py \
        --model_path ~/tbp/results/monty/pretrained_models/pretrained_janus/two_can_two_banana \
        --num_rotations 4 --seed 0
"""

import argparse
import copy
from pathlib import Path

import ablation_utils as utils
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch_geometric.data import Data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create rotated variants of pretrained Monty object models."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=utils.DEFAULT_MODEL_PATH,
        help="Directory of the pretrained model (containing pretrained/model.pt).",
    )
    parser.add_argument(
        "--num_rotations",
        type=int,
        default=4,
        help="Number of random rotation variants to create per object.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the rotations.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save the rotated model. Defaults to a sibling of "
        "model_path with the number of rotations in the name.",
    )
    return parser.parse_args()


def random_rotation(rng):
    """Sample a uniformly random rotation from a numpy Generator."""
    quat = rng.normal(size=4)
    return Rotation.from_quat(quat / np.linalg.norm(quat))


def rotate_graph(graph, rot_matrix):
    """Build a copy of `graph` rotated by `rot_matrix` about its centroid.

    Returns:
        Tuple of (rotated graph, centroid used as the rotation center).
    """
    pos = np.array(graph.pos)
    centroid = pos.mean(axis=0)
    new_pos = (pos - centroid) @ rot_matrix.T + centroid
    new_norm = np.array(graph.norm) @ rot_matrix.T

    new_x = np.array(graph.x).copy()
    feature_mapping = copy.deepcopy(graph.feature_mapping)
    if "pose_vectors" in feature_mapping:
        pv_cols = feature_mapping["pose_vectors"]
        num_nodes = len(new_x)
        pose_vectors = new_x[:, pv_cols[0] : pv_cols[1]].reshape(num_nodes, 3, 3)
        pose_vectors = pose_vectors @ rot_matrix.T
        new_x[:, pv_cols[0] : pv_cols[1]] = pose_vectors.reshape(num_nodes, 9)

    new_graph = Data(
        x=torch.tensor(new_x, dtype=torch.float),
        pos=torch.tensor(new_pos, dtype=torch.float),
        norm=torch.tensor(new_norm, dtype=torch.float),
        feature_mapping=feature_mapping,
    )
    if graph.edge_index is not None:
        new_graph.edge_index = graph.edge_index.clone()
        edge_attr = np.array(graph.edge_attr)
        if edge_attr.shape[1] == 3:
            # Displacements are pose-dependent; PPF edge attributes (4 cols)
            # are rotation-invariant and get recomputed by add_ppf_to_graph.
            edge_attr = edge_attr @ rot_matrix.T
        new_graph.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return new_graph, centroid


def plot_angle_summary(angles, save_path):
    """Save a bar chart of the rotation angle applied to each variant."""
    labels = list(angles.keys())
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, 1 + 0.8 * len(labels)), 5))
    ax.bar(x, [angles[k] for k in labels])
    for xi, label in zip(x, labels):
        ax.text(xi, angles[label], f"{angles[label]:.0f}", ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Rotation angle (deg)")
    ax.set_title("Random rotation applied per variant")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    state_dict, model_file = utils.load_state_dict(args.model_path)

    model_path = Path(args.model_path).expanduser()
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser()
    else:
        output_dir = model_path.parent / (
            f"{model_path.name}_rot{args.num_rotations}_seed{args.seed}"
        )
    _, viz_dir = utils.prepare_output_dirs(output_dir)

    rng = np.random.default_rng(args.seed)
    num_lms = len(state_dict["lm_dict"])
    variant_info = {}
    angles = {}

    for lm_id, lm_state in state_dict["lm_dict"].items():
        variants = {}
        variant_to_source = {}
        for object_id, channel_models in lm_state["graph_memory"].items():
            for i in range(1, args.num_rotations + 1):
                variant_id = f"{object_id}_rot{i}"
                rotation = random_rotation(rng)
                rot_matrix = rotation.as_matrix()
                rotvec = rotation.as_rotvec()
                angle_deg = float(np.rad2deg(np.linalg.norm(rotvec)))
                axis = (
                    rotvec / np.linalg.norm(rotvec)
                    if np.linalg.norm(rotvec) > 0
                    else np.array([1.0, 0.0, 0.0])
                )
                channel_variants = {}
                for input_channel, model in channel_models.items():
                    key = utils.display_key(
                        lm_id, variant_id, num_lms, input_channel, len(channel_models)
                    )
                    new_graph, centroid = rotate_graph(model._graph, rot_matrix)
                    channel_variants[input_channel] = utils.make_variant_model(
                        model, variant_id, new_graph
                    )
                    variant_info[key] = {
                        "source_object": object_id,
                        "lm_id": lm_id,
                        "input_channel": input_channel,
                        "num_nodes": len(new_graph.pos),
                        "quaternion_xyzw": rotation.as_quat(),
                        "rotation_axis": axis,
                        "rotation_angle_deg": angle_deg,
                        "euler_xyz_deg": rotation.as_euler("xyz", degrees=True),
                        "centroid": centroid,
                    }
                    angles[key] = angle_deg
                    utils.plot_panels(
                        f"{key}: rotated {angle_deg:.1f} deg about the centroid",
                        [
                            {
                                "title": "Before",
                                "point_sets": [{"pos": np.array(model._graph.pos)}],
                            },
                            {
                                "title": f"After: {angle_deg:.1f} deg",
                                "point_sets": [{"pos": np.array(new_graph.pos)}],
                            },
                        ],
                        viz_dir / f"{key}.png",
                    )
                    print(f"  {key}: rotated {angle_deg:.1f} deg about {np.round(axis, 3).tolist()}")
                variants[variant_id] = channel_variants
                variant_to_source[variant_id] = object_id
        utils.replace_graph_memory_with_variants(lm_state, variants, variant_to_source)

    plot_angle_summary(angles, viz_dir / "rotation_angles.png")

    g2t, t2g = utils.collect_mappings(state_dict)
    params = {
        "script": Path(__file__).name,
        "source_model": str(model_file),
        "args": vars(args),
        "variants": variant_info,
        "skipped_variants": {},
        "graph_id_to_target": g2t,
        "target_to_graph_id": t2g,
    }
    utils.save_outputs(state_dict, params, output_dir)


if __name__ == "__main__":
    main()
