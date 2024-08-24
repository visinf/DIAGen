from semantic_aug.augmentations.textual_inversion import TextualInversion
from diffusers import StableDiffusionPipeline
from itertools import product
from torch import autocast
from PIL import Image

from tqdm import trange
import os
import torch
import argparse
import numpy as np
import random


if __name__ == "__main__":
    '''
    Skript to manually generate images of the classes using a prompt with '<class_name>' as pseudo word
    
    example call from terminal:
    python test_image_generation.py --prompt "A photo of a <bench>" --num-generate 5 --embed-path "tokens/custom_coco-tokens/custom_coco-0-2.pt"
    '''

    parser = argparse.ArgumentParser("Stable Diffusion inference script")

    parser.add_argument("--model-path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--embed-path", type=str, default="tokens/custom_coco-tokens/custom_coco-0-2.pt")
    # Path to the tokens (learned pseudo words) you want to use.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-generate", type=int, default=10)

    parser.add_argument("--prompt", nargs='+', type=str, default="a photo of a <bench>")
    # The algorithm searches in the provided tokens for your provided learned representation of e.g. <bench>.
    # Attention, if the pseudo word is not found in the tokens, the text is simply passed to the tokenizer with the
    # brackets. If multiple prompts are given, for each guiding image one of them is chosen. To make it reproducible
    # the selecting process iterates through the list of prompts

    parser.add_argument("--out", type=str, default="test_generations")

    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--erasure-ckpt-name", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    torch.cuda.set_device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path, use_auth_token=True,
        revision="fp16",
        torch_dtype=torch.float16
    ).to('cuda')

    aug = TextualInversion(args.embed_path, model_path=args.model_path)
    pipe.tokenizer = aug.pipe.tokenizer
    pipe.text_encoder = aug.pipe.text_encoder

    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None

    prompt_idx = 0
    for idx in trange(args.num_generate, desc="Generating Images"):
        with autocast('cuda'):
            # Calculate the index for the current prompt (cycle through list)
            prompt_idx = idx % len(args.prompt)
            print(f"prompts_idx: {prompt_idx} -> chosen prompt: {args.prompt[prompt_idx]}")

            image = pipe(
                args.prompt[prompt_idx],
                guidance_scale=args.guidance_scale
            ).images[0]

        image.save(os.path.join(args.out, f"{idx}.png"))
