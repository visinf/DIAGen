import torch
import os
import glob
import argparse
import numpy as np
from itertools import product

DEFAULT_EMBED_PATH = "{dataset}-tokens/{dataset}-{seed}-{examples_per_class}.pt"
DEFAULT_NOISE_EMBED_PATH = "{dataset}-tokens/noise/{dataset}-{seed}-{examples_per_class}.pt"

if __name__ == "__main__":
    '''
    Step 2:
    Combines all learned embeddings into tokens. Also adds tokens with noise if this is set in the params.

    example call from terminal:
    python aggregate_embeddings.py --seeds 0 --examples-per-class 8 --dataset "coco" --augment-embeddings True --input-path "./fine-tuned" --embed-path "tokens/{dataset}-tokens/noise/{dataset}-{seed}-{examples_per_class}.pt"
    '''

    parser = argparse.ArgumentParser("Merge token files")

    # Replaced --num-trials with --seeds. To enable custom seed setting
    # parser.add_argument("--num-trials", type=int, default=8)
    parser.add_argument("--seeds", nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument("--examples-per-class", nargs='+', type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--embed-path", type=str, default=DEFAULT_EMBED_PATH)
    parser.add_argument("--input-path", type=str, default="./fine-tuned")
    parser.add_argument("--dataset", type=str, default="coco",
                        choices=["spurge", "imagenet", "coco", "pascal", "road_sign", "coco_extension", "focus"])
    parser.add_argument("--augment-embeddings", default=False, help="Whether to augment the embeddings")
    parser.add_argument("--std-deviation", nargs='+', type=float, default=[0.005, 0.01, 0.025], help="How much std-dev to use")

    args = parser.parse_args()

    for seed, examples_per_class in product(args.seeds, args.examples_per_class):

        path = os.path.join(args.input_path, f"{args.dataset}-{seed}-{examples_per_class}/*/learned_embeds.bin")

        merged_dict = dict()
        for file in glob.glob(path):
            merged_dict.update(torch.load(file))

        if args.augment_embeddings:
            merged_dict_2 = merged_dict.copy()
            for key, original_tensor in merged_dict.items():
                noise_tensors = [torch.randn_like(original_tensor) for _ in range(len(args.std_deviation))]
                std_dev_tensors = [tensor * std_dev for tensor, std_dev in zip(noise_tensors, args.std_deviation)]
                augmented_tensors = [original_tensor + std_dev_tensor for std_dev_tensor in std_dev_tensors]
                base_key = key[key.find("<") + 1:key.find(">")]
                merged_dict_2.update(
                    {f"<{base_key}_{aug}>": tensor for aug, tensor in zip(args.std_deviation, augmented_tensors)})
            merged_dict = merged_dict_2

        target_path = args.embed_path.format(dataset=args.dataset, seed=seed, examples_per_class=examples_per_class)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        torch.save(merged_dict, target_path)