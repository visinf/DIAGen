from semantic_aug.generative_augmentation import GenerativeAugmentation
from models.filter_model import ClassificationFilterModel
from typing import Tuple
from torch.utils.data import Dataset
from collections import defaultdict
from itertools import product
from tqdm import tqdm
from PIL import Image

import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import numpy as np
import abc
import os
import shutil
import copy

# DEFAULT_EMBED_PATH = "{dataset}-tokens/{dataset}-{seed}-{examples_per_class}.pt"
DEFAULT_EMBED_PATH = "coco_extension-tokens.pt"
DEFAULT_PROMPT_PATH = "prompts/prompts.csv"
DEFAULT_PROMPT = "a photo of a {name}"


class FewShotDataset(Dataset):
    """
    This dataset simulates a few-shot use case. All inherited classes use examples_per_class
    as an upper bound for the number of fixed examples they contain.
    """

    num_classes: int = None
    class_names: int = None

    def __init__(self, examples_per_class: int = None,
                 generative_aug: GenerativeAugmentation = None,
                 synthetic_probability: float = 0.5,
                 synthetic_dir: str = None,
                 use_synthetic_filter: bool = False,
                 use_llm_prompt: bool = False,
                 prompt_path: str = DEFAULT_PROMPT_PATH,
                 embed_path: str = DEFAULT_EMBED_PATH,
                 use_embedding_noise: bool = False):

        self.examples_per_class = examples_per_class
        self.generative_aug = generative_aug

        self.synthetic_probability = synthetic_probability

        self.use_synthetic_filter = use_synthetic_filter
        self.synthetic_dir = synthetic_dir
        self.synthetic_examples = defaultdict(list)
        self.synthetic_weights = defaultdict(list)

        self.use_llm_prompt = use_llm_prompt
        if prompt_path is None:
            prompt_path = DEFAULT_PROMPT_PATH
        self.prompt_path = prompt_path
        # print(f"prompt path used: {self.prompt_path}")

        self.use_embedding_noise = use_embedding_noise
        if embed_path is None:
            embed_path = DEFAULT_EMBED_PATH
        self.embed_noise_path = embed_path

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

        if synthetic_dir is not None:
            # Remove the directory and its contents and create a new one (important if trials > 1 and filter is used)
            shutil.rmtree(synthetic_dir, ignore_errors=True)
            os.makedirs(synthetic_dir)

    @abc.abstractmethod
    def get_image_by_idx(self, idx: int) -> Image.Image:

        return NotImplemented

    @abc.abstractmethod
    def get_label_by_idx(self, idx: int) -> int:

        return NotImplemented

    @abc.abstractmethod
    def get_metadata_by_idx(self, idx: int) -> dict:

        return NotImplemented

    def read_prompts_from_csv(self):
        prompts_dict = {}

        with open(self.prompt_path, mode='r', newline='', encoding='utf-8') as file:
            next(file)  # Skip the header line
            for line in file:
                row = line.strip().split(';')
                if row[0] not in prompts_dict:
                    prompts_dict[row[0]] = []
                prompts_dict[row[0]].append(row[2])

        return prompts_dict

    def read_names_from_pt(self):
        noise_name_dict = {}
        pt_content = torch.load(self.embed_noise_path)

        # fill the dict with class_names as keys and list of class_name with noise as values
        # e.g.: "bench": ["bench", "bench_0.05", ...]
        for idx in range(len(self)):
            class_name = self.get_metadata_by_idx(idx)['name']
            formated_class_name = class_name.replace(' ', '_')
            noise_name_dict[class_name] = [name[1:-1] for name in pt_content.keys() if formated_class_name in name]

        return noise_name_dict

    def load_filter(self, path: str):
        if self.use_synthetic_filter:
            state_dict = torch.load(path)
            saved_num_classes = int(state_dict["num_classes"].item())
            self.filter_model = ClassificationFilterModel(saved_num_classes)
            self.filter_model.load_state_dict(state_dict)
            self.filter_model.cuda()
            self.filter_model.eval()

    def normalize_weights(self):
        if self.use_synthetic_filter:
            for idx in range(len(self.synthetic_weights)):
                self.synthetic_weights[idx] = [w / sum(self.synthetic_weights[idx]) for w in self.synthetic_weights[idx]]

    def generate_augmentations(self, num_repeats: int):

        self.synthetic_examples.clear()
        self.synthetic_weights.clear()
        options = product(range(len(self)), range(num_repeats))

        prompts_dict = {}
        if self.use_llm_prompt:
            prompts_dict = self.read_prompts_from_csv()
            print(f"first class of prompts dict (read from csv): {prompts_dict[list(prompts_dict)[0]]}")
        class_occur = {}

        noise_name_dict = {}
        if self.use_embedding_noise:
            noise_name_dict = self.read_names_from_pt()
            print(f"use those embedding names (and gaussian scales): {noise_name_dict}")

        for idx, num in tqdm(list(
                options), desc="Generating Augmentations"):

            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)
            metadata = self.get_metadata_by_idx(idx)

            class_name = metadata['name']
            if class_name not in class_occur:
                class_occur[class_name] = -1
            class_occur[class_name] += 1

            if self.use_llm_prompt:
                # This chooses a prompt out of the list according to the occurrence of the class name
                prompt_idx = class_occur[class_name] % len(prompts_dict[class_name])
                self.generative_aug.set_augs_prompt(prompts_dict[class_name][prompt_idx])

            if self.use_embedding_noise:
                # This chooses an embed noise vector out of the list according to the occurrence of the class name
                embed_name_idx = class_occur[class_name] % len(noise_name_dict[class_name])
                metadata['name'] = noise_name_dict[class_name][embed_name_idx]

            image, label = self.generative_aug(
                image, label, metadata)

            if self.synthetic_dir is not None:
                pil_image = image  # type: PIL.Image.Image

                if self.use_synthetic_filter:
                    num_participants_in_majority_vote = 5  # Present augmented variants of the image 5 times and
                    # calculate the mean over the filter logits
                    probabilities_array_mean = []
                    with torch.no_grad():
                        for _ in range(num_participants_in_majority_vote):
                            # Add an extra batch dimension as the model expects a batch of images and change device
                            transformed_image = copy.deepcopy(image)
                            # use self.transform of subclass (e.g. road_sign.py)
                            transformed_image = self.transform(transformed_image).unsqueeze(0).cuda()
                            # Run image through model
                            logits = self.filter_model(transformed_image)
                            # Look at calibrated logits with temperature scaling
                            logits = self.filter_model.temperature_scale(logits)
                            # Apply softmax activation to convert logits into probabilities
                            probabilities = F.softmax(logits, dim=1)
                            probabilities_array = probabilities.cpu().detach().numpy()[0]
                            # Write in probabilities_array_mean
                            probabilities_array_mean.append(probabilities_array)

                        # Convert the list of arrays to a np.array
                        probabilities_array_mean = np.array(probabilities_array_mean)
                        # Calculate the mean of the arrays
                        mean_probabilities = np.mean(probabilities_array_mean, axis=0)

                        # Confidence in [0,1] that the image is correctly labeled
                        confidence = mean_probabilities[label]

                        confidence_clip = 0.5  # Images with confidence values higher than that are fully trusted
                        weight = 1 if confidence >= confidence_clip else 2 * confidence
                        self.synthetic_weights[idx].append(weight)

                    print_decision = False
                    if print_decision:
                        print(f'Image: label_{label}-{idx}-{num}.png')
                        predicted_class = np.argmax(mean_probabilities)
                        print(f'Highest class: {predicted_class} with probability of: '
                              f'{np.round(mean_probabilities[predicted_class], 3)}')
                        if not np.isclose(predicted_class, label):
                            print(f'Different classified, probability of given label {label}: '
                                  f'{np.round(mean_probabilities[label], 3)}')
                        print(f'Weight: {weight}')

                image_path = os.path.join(self.synthetic_dir, f"label_{label}-{idx}-{num}.png")
                self.synthetic_examples[idx].append((image_path, label))
                pil_image.save(image_path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        if len(self.synthetic_examples[idx]) > 0 and \
                np.random.uniform() < self.synthetic_probability:

            # Extract all synthetic images and labels from the list of tuples
            images, labels = zip(*self.synthetic_examples[idx])
            # Select an image based on a weighted distribution
            if self.use_synthetic_filter:
                image = np.random.choice(images, p=self.synthetic_weights[idx])
            else:
                image = np.random.choice(images)
            label = labels[0]
            if isinstance(image, str):
                image = Image.open(image)

        else:
            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)

        return self.transform(image), label
