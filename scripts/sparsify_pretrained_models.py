# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Create sparsified variants of pretrained Monty object models.

For every object model stored in a pretrained `model.pt` and every
requested sparsity level, this script creates a variant object (e.g.
`banana_1_sparsity40`) that keeps that percentage of the learned graph
nodes, sampled uniformly at random. The output model directory contains
only the variant objects; the id mappings are rewritten so a detected
variant still counts as the original target object.

Example:
    python scripts/sparsify_pretrained_models.py \
        --model_path ~/tbp/results/monty/pretrained_models/pretrained_janus/two_can_two_banana \
        --sparsity 20 40 60 80 --seed 0
"""

import argparse
from pathlib import Path

import ablation_utils as utils
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create sparsified variants of pretrained Monty object models."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=utils.DEFAULT_MODEL_PATH,
        help="Directory of the pretrained model (containing pretrained/model.pt).",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        nargs="+",
        default=[20, 40, 60, 80],
        help="Percentages of learned points to keep; one variant per value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the node sampling.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save the sparsified model. Defaults to a sibling of "
        "model_path with the sparsity levels in the name.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    state_dict, model_file = utils.load_state_dict(args.model_path)

    model_path = Path(args.model_path).expanduser()
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser()
    else:
        levels = "-".join(f"{s:g}" for s in args.sparsity)
        output_dir = model_path.parent / (
            f"{model_path.name}_sparsity{levels}_seed{args.seed}"
        )
    _, viz_dir = utils.prepare_output_dirs(output_dir)

    rng = np.random.default_rng(args.seed)
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
            for sparsity in args.sparsity:
                variant_id = f"{object_id}_sparsity{sparsity:g}"
                channel_variants = {}
                for input_channel, model in channel_models.items():
                    key = utils.display_key(
                        lm_id, variant_id, num_lms, input_channel, len(channel_models)
                    )
                    num_nodes = len(model.pos)
                    baseline[object_key] = num_nodes
                    num_keep = int(round(num_nodes * sparsity / 100))
                    clamped = False
                    if num_keep < 3:
                        if num_nodes < 3:
                            skipped[key] = {
                                "reason": "source model has fewer than 3 nodes",
                                "num_nodes": num_nodes,
                            }
                            print(f"  {key}: skipped ({num_nodes} nodes in source)")
                            continue
                        num_keep = 3
                        clamped = True
                    keep_ids = np.sort(
                        rng.choice(num_nodes, size=num_keep, replace=False)
                    )
                    new_graph = utils.subset_graph(model._graph, keep_ids, model.k_n)
                    channel_variants[input_channel] = utils.make_variant_model(
                        model, variant_id, new_graph
                    )
                    variant_info[key] = {
                        "source_object": object_id,
                        "lm_id": lm_id,
                        "input_channel": input_channel,
                        "sparsity_percent": sparsity,
                        "num_nodes_before": int(num_nodes),
                        "num_nodes_after": int(num_keep),
                        "clamped": clamped,
                    }
                    counts[object_key][f"{sparsity:g}%"] = int(num_keep)
                    utils.plot_panels(
                        f"{key}: {sparsity:g}% of learned points kept",
                        [
                            {
                                "title": f"Before: {num_nodes} points",
                                "point_sets": [{"pos": np.array(model._graph.pos)}],
                            },
                            {
                                "title": f"After: {num_keep} points",
                                "point_sets": [{"pos": np.array(new_graph.pos)}],
                            },
                        ],
                        viz_dir / f"{key}.png",
                    )
                    print(f"  {key}: {num_nodes} -> {num_keep} points")
                if channel_variants:
                    variants[variant_id] = channel_variants
                    variant_to_source[variant_id] = object_id
        utils.replace_graph_memory_with_variants(lm_state, variants, variant_to_source)

    utils.plot_variant_count_summary(
        counts,
        viz_dir / "node_counts.png",
        "Learned points per object at each sparsity level",
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
