# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Split pretrained Monty object models into per-viewpoint variants.

For every object model stored in a pretrained `model.pt` and every
requested azimuth, this script creates a variant object (e.g.
`banana_1_view20`) containing only the graph nodes visible from a camera
at that azimuth. Cameras sit on a ring around the vertical (y) axis
through the object's centroid, at a configurable elevation angle and at a
distance scaled from the object's bounding radius. Azimuth 0 looks from
the +z direction; positive azimuths rotate toward +x.

Visibility is computed with hidden point removal (Katz et al.): points
are spherically flipped about the camera and the visible ones are those
on the convex hull of the flipped cloud. Kept nodes retain their original
3D coordinates; node ids are renumbered and k-NN edges rebuilt. The
output model directory contains only the variant objects; the id mappings
are rewritten so a detected variant still counts as the original target.

Example:
    python scripts/viewpoint_split_pretrained_models.py \
        --model_path ~/tbp/results/monty/pretrained_models/pretrained_janus/two_can_two_banana \
        --viewpoints 0 20 40 --elevation_deg 30
"""

import argparse
from pathlib import Path

import ablation_utils as utils
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split pretrained Monty object models into viewpoint variants."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=utils.DEFAULT_MODEL_PATH,
        help="Directory of the pretrained model (containing pretrained/model.pt).",
    )
    parser.add_argument(
        "--viewpoints",
        type=float,
        nargs="+",
        default=[0, 20, 40],
        help="Camera azimuths in degrees around the vertical axis; one "
        "variant per value.",
    )
    parser.add_argument(
        "--elevation_deg",
        type=float,
        default=30.0,
        help="Camera elevation above the horizontal plane in degrees.",
    )
    parser.add_argument(
        "--distance_multiplier",
        type=float,
        default=3.0,
        help="Camera distance as a multiple of the object's bounding radius.",
    )
    parser.add_argument(
        "--hpr_gamma",
        type=float,
        default=2.0,
        help="Exponent of the hidden-point-removal sphere radius "
        "(radius = max distance * 10^gamma). Larger keeps more points.",
    )
    parser.add_argument(
        "--min_visible_nodes",
        type=int,
        default=3,
        help="Skip a variant if fewer nodes than this are visible.",
    )
    parser.add_argument(
        "--normal_culling",
        action="store_true",
        help="Additionally drop nodes whose stored surface normal faces "
        "away from the camera.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save the split model. Defaults to a sibling of "
        "model_path with the viewpoint parameters in the name.",
    )
    return parser.parse_args()


def camera_position(centroid, bounding_radius, azimuth_deg, elevation_deg, multiplier):
    """Camera position on a ring around the vertical axis through `centroid`."""
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    direction = np.array(
        [
            np.sin(azimuth) * np.cos(elevation),
            np.sin(elevation),
            np.cos(azimuth) * np.cos(elevation),
        ]
    )
    return centroid + multiplier * bounding_radius * direction


def main():
    args = parse_args()
    state_dict, model_file = utils.load_state_dict(args.model_path)

    model_path = Path(args.model_path).expanduser()
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser()
    else:
        views = "-".join(f"{v:g}" for v in args.viewpoints)
        name = (
            f"{model_path.name}_views{views}"
            f"_elev{args.elevation_deg:g}_dist{args.distance_multiplier:g}"
        )
        if args.hpr_gamma != 2.0:
            name += f"_gamma{args.hpr_gamma:g}"
        output_dir = model_path.parent / name
    _, viz_dir = utils.prepare_output_dirs(output_dir)

    num_lms = len(state_dict["lm_dict"])
    variant_info = {}
    skipped = {}
    counts = {}
    baseline = {}

    for lm_id, lm_state in state_dict["lm_dict"].items():
        variants = {}
        variant_to_source = {}
        for object_id, channel_models in lm_state["graph_memory"].items():
            object_key = utils.display_key(lm_id, object_id, num_lms)
            counts.setdefault(object_key, {})
            all_pos = np.vstack(
                [np.array(m.pos) for m in channel_models.values()]
            )
            centroid = all_pos.mean(axis=0)
            bounding_radius = float(np.linalg.norm(all_pos - centroid, axis=1).max())
            for azimuth in args.viewpoints:
                variant_id = f"{object_id}_view{azimuth:g}"
                camera = camera_position(
                    centroid,
                    bounding_radius,
                    azimuth,
                    args.elevation_deg,
                    args.distance_multiplier,
                )
                channel_variants = {}
                for input_channel, model in channel_models.items():
                    key = utils.display_key(
                        lm_id, variant_id, num_lms, input_channel, len(channel_models)
                    )
                    pos = np.array(model.pos)
                    num_nodes = len(pos)
                    baseline[object_key] = num_nodes
                    visible, fallback = utils.hidden_point_removal(
                        pos, camera, gamma=args.hpr_gamma
                    )
                    if args.normal_culling:
                        norm = np.array(model.norm)
                        facing = np.einsum("ij,ij->i", norm, camera - pos) > 0
                        visible = visible[facing[visible]]
                    num_visible = len(visible)
                    if num_visible < args.min_visible_nodes:
                        skipped[key] = {
                            "reason": "too few visible nodes",
                            "num_visible": int(num_visible),
                        }
                        print(f"  {key}: skipped ({num_visible} visible nodes)")
                        continue
                    new_graph = utils.subset_graph(model._graph, visible, model.k_n)
                    channel_variants[input_channel] = utils.make_variant_model(
                        model, variant_id, new_graph
                    )
                    variant_info[key] = {
                        "source_object": object_id,
                        "lm_id": lm_id,
                        "input_channel": input_channel,
                        "azimuth_deg": azimuth,
                        "elevation_deg": args.elevation_deg,
                        "camera_position": camera,
                        "camera_distance": args.distance_multiplier * bounding_radius,
                        "bounding_radius": bounding_radius,
                        "hpr_gamma": args.hpr_gamma,
                        "hpr_fallback": fallback,
                        "normal_culling": args.normal_culling,
                        "num_nodes_before": int(num_nodes),
                        "num_nodes_after": int(num_visible),
                        "visible_fraction": float(num_visible / num_nodes),
                    }
                    counts[object_key][f"view{azimuth:g}"] = int(num_visible)
                    removed = np.setdiff1d(np.arange(num_nodes), visible)
                    utils.plot_panels(
                        f"{key}: {num_visible}/{num_nodes} nodes visible from "
                        f"azimuth {azimuth:g} deg",
                        [
                            {
                                "title": f"Visible: {num_visible}, "
                                f"hidden: {len(removed)}",
                                "point_sets": [
                                    {"pos": pos[visible], "label": "visible"},
                                    {
                                        "pos": pos[removed],
                                        "color": "gray",
                                        "alpha": 0.15,
                                        "label": "hidden",
                                    },
                                ],
                                "camera": camera,
                            },
                        ],
                        viz_dir / f"{key}.png",
                    )
                    print(
                        f"  {key}: {num_visible}/{num_nodes} nodes visible "
                        f"({100 * num_visible / num_nodes:.1f}%)"
                    )
                if channel_variants:
                    variants[variant_id] = channel_variants
                    variant_to_source[variant_id] = object_id
        utils.replace_graph_memory_with_variants(lm_state, variants, variant_to_source)

    utils.plot_variant_count_summary(
        counts,
        viz_dir / "node_counts.png",
        "Visible points per object at each viewpoint",
        baseline=baseline,
    )

    g2t, t2g = utils.collect_mappings(state_dict)
    params = {
        "script": Path(__file__).name,
        "source_model": str(model_file),
        "args": vars(args),
        "variants": variant_info,
        "skipped_variants": skipped,
        "graph_id_to_target": g2t,
        "target_to_graph_id": t2g,
    }
    utils.save_outputs(state_dict, params, output_dir)


if __name__ == "__main__":
    main()
