# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Shared helpers for the model ablation scripts in this directory.

The ablation scripts load a pretrained Monty `model.pt`, derive variant
copies of every object graph (sparsified, rotated, or split into
viewpoints), and save a new pretrained model directory that contains only
the variants. This module holds the pieces they share: state dict IO,
graph subsetting, edge rebuilding, graph-memory replacement with correct
`target_to_graph_id`/`graph_id_to_target` rewrites, hidden point removal,
and plotting.
"""

import copy
import json
import sys
from pathlib import Path

# Make sure the checkout this script lives in is the version of tbp.monty that
# gets imported (needed to unpickle GraphObjectModel with the right code).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
from scipy.spatial import ConvexHull, QhullError
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

from tbp.monty.frameworks.utils.graph_matching_utils import get_correct_k_n

DEFAULT_MODEL_PATH = str(
    Path.home()
    / "tbp/results/monty/pretrained_models/pretrained_janus/two_can_two_banana"
)


def load_state_dict(model_path):
    """Load a pretrained state dict from a model directory.

    Returns:
        Tuple of (state dict, resolved path of the loaded `model.pt`).
    """
    model_path = Path(model_path).expanduser()
    model_file = model_path / "pretrained" / "model.pt"
    if not model_file.exists():
        model_file = model_path / "model.pt"
    print(f"Loading pretrained model from {model_file}")
    return torch.load(model_file, weights_only=False), model_file


def prepare_output_dirs(output_dir):
    """Create `pretrained` and `visualizations` subdirectories.

    Returns:
        Tuple of (pretrained dir, visualizations dir).
    """
    pretrained_dir = output_dir / "pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    return pretrained_dir, viz_dir


def rebuild_edges(graph, positions, k_n):
    """Recompute k-NN edges and displacement edge attributes for a graph.

    Mirrors GraphObjectModel._build_adjacency_graph edge construction so the
    modified graph stays consistent with how Monty builds graphs.
    """
    k_n = get_correct_k_n(k_n, len(positions))
    scipy_graph = kneighbors_graph(positions, n_neighbors=k_n, include_self=False)
    edge_index = torch_geometric.utils.from_scipy_sparse_matrix(scipy_graph)[0]
    displacements = positions[edge_index[1]] - positions[edge_index[0]]
    graph.edge_index = edge_index
    graph.edge_attr = torch.tensor(displacements, dtype=torch.float)


def subset_graph(graph, keep_ids, k_n):
    """Build a new graph containing only the nodes in `keep_ids`.

    Node ids are renumbered to `0..len(keep_ids)-1` and k-NN edges are
    rebuilt if the source graph had edges.

    Returns:
        The new graph, or None if fewer than 3 nodes are kept (too few to
        build a valid k-NN graph).
    """
    num_keep = len(keep_ids)
    if num_keep <= 2:
        return None
    new_pos = np.array(graph.pos)[keep_ids]
    new_norm = np.array(graph.norm)[keep_ids]
    new_x = np.array(graph.x)[keep_ids].copy()
    feature_mapping = copy.deepcopy(graph.feature_mapping)
    node_id_cols = feature_mapping["node_ids"]
    new_x[:, node_id_cols[0] : node_id_cols[1]] = np.arange(num_keep).reshape(-1, 1)
    new_graph = Data(
        x=torch.tensor(new_x, dtype=torch.float),
        pos=torch.tensor(new_pos, dtype=torch.float),
        norm=torch.tensor(new_norm, dtype=torch.float),
        feature_mapping=feature_mapping,
    )
    if graph.edge_index is not None:
        rebuild_edges(new_graph, new_pos, k_n)
    return new_graph


def make_variant_model(source_model, variant_id, new_graph):
    """Create a variant copy of `source_model` holding `new_graph`.

    Returns:
        The new GraphObjectModel.
    """
    model = copy.deepcopy(source_model)
    model.object_id = variant_id
    model.set_graph(new_graph)
    if model.has_ppf:
        model.add_ppf_to_graph()
    return model


def replace_graph_memory_with_variants(lm_state, variants, variant_to_source):
    """Replace an LM's graph memory with variant models.

    Rewrites `target_to_graph_id` and `graph_id_to_target` so each variant
    maps to its source object's targets. This keeps performance
    classification working: a detected variant graph still counts as
    correct for the original target object.
    """
    old_g2t = lm_state["graph_id_to_target"]
    old_t2g = lm_state["target_to_graph_id"]
    lm_state["graph_memory"] = variants
    lm_state["graph_id_to_target"] = {
        variant_id: set(old_g2t[source])
        for variant_id, source in variant_to_source.items()
    }
    new_t2g = {}
    for target, graph_ids in old_t2g.items():
        variant_ids = {
            variant_id
            for variant_id, source in variant_to_source.items()
            if source in graph_ids
        }
        if variant_ids:
            new_t2g[target] = variant_ids
    lm_state["target_to_graph_id"] = new_t2g


def display_key(lm_id, name, num_lms, input_channel=None, num_channels=1):
    """Build a unique display key for logs, JSON entries, and filenames."""
    key = name if num_lms == 1 else f"lm{lm_id}_{name}"
    if num_channels > 1:
        key = f"{key}_{input_channel}"
    return key


def hidden_point_removal(points, camera, gamma=2.0):
    """Indices of points visible from `camera` (Katz et al. spherical flip).

    Points are reflected about a sphere of radius `max_dist * 10**gamma`
    centered on the camera; points on the convex hull of the flipped cloud
    plus the camera are visible.

    Returns:
        Tuple of (sorted array of visible indices, whether the all-visible
        fallback was used because the convex hull failed).
    """
    num_points = len(points)
    if num_points < 4:
        return np.arange(num_points), False
    p = points - camera
    dist = np.maximum(np.linalg.norm(p, axis=1), 1e-12)
    radius = dist.max() * 10.0**gamma
    flipped = p + 2 * (radius - dist)[:, None] * p / dist[:, None]
    hull_points = np.vstack([flipped, np.zeros(3)])
    try:
        hull = ConvexHull(hull_points)
    except QhullError:
        try:
            hull = ConvexHull(hull_points, qhull_options="QJ")
        except QhullError:
            return np.arange(num_points), True
    visible = np.sort(np.array([v for v in hull.vertices if v < num_points]))
    return visible, False


def plot_panels(title, panels, save_path):
    """Save a row of 3D scatter panels with shared cubic axis limits.

    Each panel is a dict with `title`, `point_sets` (list of dicts holding
    `pos` plus scatter kwargs), and an optional `camera` position drawn as
    a star with a dashed line to the first point set's centroid.
    """
    fig = plt.figure(figsize=(5.5 * len(panels), 5))
    all_points = []
    for panel in panels:
        all_points.extend(ps["pos"] for ps in panel["point_sets"])
        if panel.get("camera") is not None:
            all_points.append(np.atleast_2d(panel["camera"]))
    all_points = np.vstack(all_points)
    center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
    half_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 2 * 1.1

    for i, panel in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
        for point_set in panel["point_sets"]:
            kwargs = {k: v for k, v in point_set.items() if k != "pos"}
            kwargs.setdefault("s", 3)
            kwargs.setdefault("alpha", 0.6)
            pos = point_set["pos"]
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], **kwargs)
        camera = panel.get("camera")
        if camera is not None:
            centroid = panel["point_sets"][0]["pos"].mean(axis=0)
            ax.scatter(*camera, marker="*", s=150, color="red", label="camera")
            ax.plot(
                *zip(camera, centroid), linestyle="--", color="gray", linewidth=1
            )
        if any("label" in ps for ps in panel["point_sets"]) or camera is not None:
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(panel["title"])
        for dim, setter in zip(center, (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            setter(dim - half_range, dim + half_range)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_variant_count_summary(counts, save_path, title, baseline=None):
    """Save a grouped bar chart of node counts per object and variant.

    `counts` maps object key -> {variant label -> node count}; `baseline`
    optionally maps object key -> original node count.
    """
    objects = list(counts.keys())
    labels = list(counts[objects[0]].keys())
    series = []
    if baseline is not None:
        series.append(("original", [baseline[o] for o in objects]))
    for label in labels:
        series.append((label, [counts[o].get(label, 0) for o in objects]))

    x = np.arange(len(objects))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(max(8, 2 + 2.5 * len(objects)), 5))
    for j, (label, values) in enumerate(series):
        offsets = x + (j - (len(series) - 1) / 2) * width
        ax.bar(offsets, values, width=width, label=label)
        for xi, value in zip(offsets, values):
            ax.text(xi, value, str(value), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(objects, rotation=20, ha="right")
    ax.set_ylabel("Number of learned points")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _json_default(obj):
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def collect_mappings(state_dict):
    """Gather the id mappings of all LMs for the JSON summary.

    Returns:
        Tuple of (graph_id_to_target, target_to_graph_id), keys prefixed
        with the LM id when there is more than one LM.
    """
    num_lms = len(state_dict["lm_dict"])
    g2t, t2g = {}, {}
    for lm_id, lm_state in state_dict["lm_dict"].items():
        for graph_id, targets in lm_state["graph_id_to_target"].items():
            g2t[display_key(lm_id, graph_id, num_lms)] = targets
        for target, graph_ids in lm_state["target_to_graph_id"].items():
            t2g[display_key(lm_id, target, num_lms)] = graph_ids
    return g2t, t2g


def save_outputs(state_dict, params, output_dir):
    """Save the modified state dict and the JSON parameter summary."""
    params_file = output_dir / "ablation_params.json"
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2, default=_json_default)
    model_file = output_dir / "pretrained" / "model.pt"
    torch.save(state_dict, model_file)
    print(f"\nSaved model to {model_file}")
    print(f"Parameters saved to {params_file}")
    print(f"Visualizations saved to {output_dir / 'visualizations'}")
