# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Create ablated copies of pretrained Monty object models.

For every object model stored in a pretrained ``model.pt`` this script:
  1. Randomly removes a proportion of the learned graph nodes (default 50%).
  2. Applies a random rigid transform (rotation + translation) to the
     remaining points, including all direction-valued features
     (surface normals and pose vectors).
  3. Rebuilds the k-nearest-neighbor edges of the graph so the ablated model
     remains internally consistent.

The modified state dict is saved as a new pretrained model directory whose
name encodes the ablation parameters, so it can be pointed to directly by an
evaluation experiment's ``model_name_or_path``. Before/after visualizations
of every object graph and a JSON summary of the parameters are stored
alongside the model.

Example:
    python ablate_pretrained_models.py \
        --model_path ~/tbp/results/monty/pretrained_models/pretrained_janus/two_can_two_banana \
        --ablate_fraction 0.5 --max_translation 0.02 --max_rotation_deg 45 --seed 0
"""

import argparse
import copy
import json
import sys
from pathlib import Path

# Make sure the checkout this script lives in is the version of tbp.monty that
# gets imported (needed to unpickle GraphObjectModel with the right code).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
from scipy.spatial.transform import Rotation
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

from tbp.monty.frameworks.utils.graph_matching_utils import get_correct_k_n


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablate learned points of pretrained Monty object models."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(
            Path.home()
            / "tbp/results/monty/pretrained_models/pretrained_janus/spoon"
        ),
        help="Directory of the pretrained model (containing pretrained/model.pt).",
    )
    parser.add_argument(
        "--ablate_fraction",
        type=float,
        default=0.75,
        help="Proportion of learned points to randomly remove from each model.",
    )
    parser.add_argument(
        "--max_translation",
        type=float,
        default=0.0,
        help="Maximum translation (meters) applied per axis, drawn uniformly "
        "from [-max_translation, max_translation].",
    )
    parser.add_argument(
        "--max_rotation_deg",
        type=float,
        default=0.0,
        help="Maximum rotation angle (degrees) about a random axis, drawn "
        "uniformly from [0, max_rotation_deg].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the ablation and transforms.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save the ablated model. Defaults to a sibling of "
        "model_path with the ablation parameters in the name.",
    )
    return parser.parse_args()


def random_rigid_transform(rng, max_translation, max_rotation_deg):
    """Sample a random rotation matrix and translation vector.

    Returns:
        Tuple of (rotation matrix (3x3), translation vector (3,), angle in
        degrees, rotation axis).
    """
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle_deg = rng.uniform(0, max_rotation_deg)
    rotation = Rotation.from_rotvec(np.deg2rad(angle_deg) * axis)
    translation = rng.uniform(-max_translation, max_translation, size=3)
    return rotation.as_matrix(), translation, angle_deg, axis


def rebuild_edges(graph, positions, k_n):
    """Recompute k-NN edges and displacement edge attributes for a graph.

    Mirrors GraphObjectModel._build_adjacency_graph edge construction so the
    ablated graph stays consistent with how Monty builds graphs.
    """
    k_n = get_correct_k_n(k_n, len(positions))
    scipy_graph = kneighbors_graph(positions, n_neighbors=k_n, include_self=False)
    edge_index = torch_geometric.utils.from_scipy_sparse_matrix(scipy_graph)[0]
    displacements = positions[edge_index[1]] - positions[edge_index[0]]
    graph.edge_index = edge_index
    graph.edge_attr = torch.tensor(displacements, dtype=torch.float)


def ablate_object_model(model, rng, args):
    """Ablate and rigidly transform a single GraphObjectModel in place.

    Returns:
        Dict summarizing the changes applied to this model.
    """
    old_graph = model._graph
    num_nodes = len(old_graph.pos)
    num_keep = max(2, int(round(num_nodes * (1 - args.ablate_fraction))))
    keep_ids = np.sort(rng.choice(num_nodes, size=num_keep, replace=False))

    rot_matrix, translation, angle_deg, axis = random_rigid_transform(
        rng, args.max_translation, args.max_rotation_deg
    )

    old_pos = np.array(old_graph.pos)
    # Rotate about the model's centroid (not the reference-frame origin, which
    # can be far from the object) so the displacement of the point cloud is
    # governed by max_translation alone.
    centroid = old_pos[keep_ids].mean(axis=0)
    new_pos = (
        (old_pos[keep_ids] - centroid) @ rot_matrix.T + centroid + translation
    )
    new_norm = np.array(old_graph.norm)[keep_ids] @ rot_matrix.T

    new_x = np.array(old_graph.x)[keep_ids].copy()
    feature_mapping = copy.deepcopy(old_graph.feature_mapping)
    # Re-number node ids so they match the new graph indices.
    node_id_cols = feature_mapping["node_ids"]
    new_x[:, node_id_cols[0] : node_id_cols[1]] = np.arange(num_keep).reshape(-1, 1)
    # Pose vectors are three 3D direction vectors per node; rotate each.
    if "pose_vectors" in feature_mapping:
        pv_cols = feature_mapping["pose_vectors"]
        pose_vectors = new_x[:, pv_cols[0] : pv_cols[1]].reshape(num_keep, 3, 3)
        pose_vectors = pose_vectors @ rot_matrix.T
        new_x[:, pv_cols[0] : pv_cols[1]] = pose_vectors.reshape(num_keep, 9)

    new_graph = Data(
        x=torch.tensor(new_x, dtype=torch.float),
        pos=torch.tensor(new_pos, dtype=torch.float),
        norm=torch.tensor(new_norm, dtype=torch.float),
        feature_mapping=feature_mapping,
    )
    if old_graph.edge_index is not None:
        rebuild_edges(new_graph, new_pos, model.k_n)
    model.set_graph(new_graph)
    if model.has_ppf:
        model.add_ppf_to_graph()

    return {
        "num_nodes_before": int(num_nodes),
        "num_nodes_after": int(num_keep),
        "rotation_angle_deg": float(angle_deg),
        "rotation_axis": [float(v) for v in axis],
        "translation_m": [float(v) for v in translation],
    }, old_pos


def plot_before_after(object_id, old_pos, new_pos, save_path):
    """Save a 3D scatter of the graph nodes before and after ablation."""
    fig = plt.figure(figsize=(11, 5))
    all_pos = np.vstack([old_pos, new_pos])
    center = (all_pos.min(axis=0) + all_pos.max(axis=0)) / 2
    half_range = (all_pos.max(axis=0) - all_pos.min(axis=0)).max() / 2 * 1.1

    for i, (pos, title) in enumerate(
        [
            (old_pos, f"Before: {len(old_pos)} points"),
            (new_pos, f"After: {len(new_pos)} points"),
        ]
    ):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=3, alpha=0.6)
        ax.set_title(title)
        for dim, setter in zip(
            center, (ax.set_xlim, ax.set_ylim, ax.set_zlim)
        ):
            setter(dim - half_range, dim + half_range)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    fig.suptitle(f"{object_id}: learned graph before vs. after ablation")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_node_count_summary(summary, save_path):
    """Save a bar chart comparing node counts before/after per object."""
    labels = list(summary.keys())
    before = [summary[k]["num_nodes_before"] for k in labels]
    after = [summary[k]["num_nodes_after"] for k in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, before, width=0.4, label="Before")
    ax.bar(x + 0.2, after, width=0.4, label="After")
    for xi, (b, a) in zip(x, zip(before, after)):
        ax.text(xi - 0.2, b, str(b), ha="center", va="bottom", fontsize=8)
        ax.text(xi + 0.2, a, str(a), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Number of learned points")
    ax.set_title("Learned points per object model before vs. after ablation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    model_path = Path(args.model_path).expanduser()
    model_file = model_path / "pretrained" / "model.pt"
    if not model_file.exists():
        model_file = model_path / "model.pt"
    print(f"Loading pretrained model from {model_file}")
    state_dict = torch.load(model_file, weights_only=False)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser()
    else:
        output_dir = model_path.parent / (
            f"{model_path.name}_ablate{args.ablate_fraction:.2f}"
            f"_trans{args.max_translation:.3f}"
            f"_rot{args.max_rotation_deg:.0f}deg"
            f"_seed{args.seed}"
        )
    (output_dir / "pretrained").mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)
    summary = {}

    for lm_id, lm_state in state_dict["lm_dict"].items():
        for object_id, channel_models in lm_state["graph_memory"].items():
            for input_channel, model in channel_models.items():
                key = (
                    object_id
                    if len(state_dict["lm_dict"]) == 1
                    else f"lm{lm_id}_{object_id}"
                )
                if len(channel_models) > 1:
                    key = f"{key}_{input_channel}"
                model_summary, old_pos = ablate_object_model(model, rng, args)
                summary[key] = model_summary
                plot_before_after(
                    key,
                    old_pos,
                    np.array(model.pos),
                    viz_dir / f"{key}.png",
                )
                print(
                    f"  {key}: {model_summary['num_nodes_before']} -> "
                    f"{model_summary['num_nodes_after']} points, rotated "
                    f"{model_summary['rotation_angle_deg']:.1f} deg, translated "
                    f"{np.round(model_summary['translation_m'], 4).tolist()} m"
                )

    plot_node_count_summary(summary, viz_dir / "node_counts.png")

    params = {
        "source_model": str(model_file),
        "ablate_fraction": args.ablate_fraction,
        "max_translation_m": args.max_translation,
        "max_rotation_deg": args.max_rotation_deg,
        "seed": args.seed,
        "per_object": summary,
    }
    with open(output_dir / "ablation_params.json", "w") as f:
        json.dump(params, f, indent=2)

    torch.save(state_dict, output_dir / "pretrained" / "model.pt")
    print(f"\nSaved ablated model to {output_dir / 'pretrained' / 'model.pt'}")
    print(f"Parameters saved to {output_dir / 'ablation_params.json'}")
    print(f"Visualizations saved to {viz_dir}")


if __name__ == "__main__":
    main()