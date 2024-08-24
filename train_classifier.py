from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.custom_coco import CustomCOCO
from semantic_aug.datasets.focus import FOCUS
from semantic_aug.augmentations.compose import ComposeParallel
from semantic_aug.augmentations.compose import ComposeSequential
from semantic_aug.augmentations.real_guidance import RealGuidance
from semantic_aug.augmentations.textual_inversion import TextualInversion
from semantic_aug.few_shot_dataset import DEFAULT_PROMPT_PATH, DEFAULT_PROMPT
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import DeiTModel
from itertools import product
from tqdm import trange
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import pandas as pd
import numpy as np
import random
import os

from train_filter import train_filter

try:
    from cutmix.cutmix import CutMix

    IS_CUTMIX_INSTALLED = True
except:
    IS_CUTMIX_INSTALLED = False

DEFAULT_MODEL_PATH = "CompVis/stable-diffusion-v1-4"

DEFAULT_DIR = "RESULTS/{dataset}_{examples_per_class}epc/{method}"

DEFAULT_EMBED_PATH = "tokens/{dataset}-tokens/{dataset}-{seed}-{examples_per_class}.pt"
DEFAULT_NOISE_EMBED_PATH = "tokens/{dataset}-tokens/noise/{dataset}-{seed}-{examples_per_class}.pt"

DATASETS = {
    "coco": COCODataset,
    "custom_coco": CustomCOCO,
    "focus": FOCUS,
}

COMPOSERS = {
    "parallel": ComposeParallel,
    "sequential": ComposeSequential
}

AUGMENTATIONS = {
    "real-guidance": RealGuidance,
    "textual-inversion": TextualInversion,
}


def run_experiment(examples_per_class: int = 0,
                   seed: int = 0,
                   dataset: str = "focus",
                   num_synthetic: int = 100,
                   iterations_per_epoch: int = 200,
                   num_epochs: int = 50,
                   batch_size: int = 32,
                   aug: List[str] = None,
                   strength: List[float] = None,
                   guidance_scale: List[float] = None,
                   mask: List[bool] = None,
                   inverted: List[bool] = None,
                   probs: List[float] = None,
                   compose: str = "parallel",
                   synthetic_probability: float = 0.5,
                   synthetic_dir: str = "synthetics",
                   embed_path: str = DEFAULT_EMBED_PATH,
                   model_path: str = DEFAULT_MODEL_PATH,
                   prompt: str = DEFAULT_PROMPT,
                   tokens_per_class: int = 4,
                   use_randaugment: bool = False,
                   use_cutmix: bool = False,
                   erasure_ckpt_path: str = None,
                   image_size: int = 256,
                   classifier_backbone: str = "resnet50",
                   synthetic_filter: str = None,
                   filter_mask_area: int = 0,
                   use_llm_prompt: bool = False,
                   prompt_path: str = DEFAULT_PROMPT_PATH,
                   save_model: bool = True,
                   eval_on_test_set: List[str] = [],
                   logdir: str = "logs",
                   use_embedding_noise: bool = False,
                   method: str = None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    use_synthetic_filter = True if synthetic_filter in ["train", "use"] else False
    if synthetic_filter == "train":
        # Initialize and train the ClassificationFilterModel here and save it in models
        train_filter(examples_per_class=examples_per_class,
                     seed=seed,
                     dataset=dataset,
                     batch_size=batch_size,
                     image_size=image_size)

    if aug is not None:
        aug = COMPOSERS[compose]([

            AUGMENTATIONS[aug](
                embed_path=embed_path,
                model_path=model_path,
                prompt=prompt,  # this is only the initialization with the default prompt
                strength=strength,
                guidance_scale=guidance_scale,
                mask=mask,
                inverted=inverted,
                erasure_ckpt_path=erasure_ckpt_path,
                tokens_per_class=tokens_per_class
            )

            for (aug, guidance_scale,
                 strength, mask, inverted) in zip(
                aug, guidance_scale,
                strength, mask, inverted
            )

        ], probs=probs)

    train_dataset = DATASETS[dataset](
        split="train", examples_per_class=examples_per_class,
        synthetic_probability=synthetic_probability,
        synthetic_dir=synthetic_dir,
        use_randaugment=use_randaugment,
        generative_aug=aug, seed=seed,
        image_size=(image_size, image_size),
        use_synthetic_filter=use_synthetic_filter,
        filter_mask_area=filter_mask_area,
        use_llm_prompt=use_llm_prompt,
        prompt_path=prompt_path,
        embed_path=embed_path,
        use_embedding_noise=use_embedding_noise)

    if num_synthetic > 0 and aug is not None:
        if use_synthetic_filter:
            train_dataset.load_filter(path=f"models/filter_{dataset}_{seed}_{examples_per_class}.pth")
        train_dataset.generate_augmentations(num_synthetic)
        if use_synthetic_filter:
            train_dataset.normalize_weights()

    cutmix_dataset = None
    if use_cutmix and IS_CUTMIX_INSTALLED:
        cutmix_dataset = CutMix(
            train_dataset, beta=1.0, prob=0.5, num_mix=2,
            num_class=train_dataset.num_classes)

    # Calculate class weights based on the inverse of class frequencies. Assign weight to each sample in the dataset
    # based on the class distribution, so that each class has an equal contribution to the overall loss.
    # If class_count is 0 set the corresponding entry in class_weights to 0 too.
    class_weights = np.where(train_dataset.class_counts == 0, 0, 1.0 / train_dataset.class_counts)
    weights = [class_weights[label] for label in train_dataset.all_labels]

    weighted_train_sampler = WeightedRandomSampler(
        weights, replacement=True,
        num_samples=batch_size * iterations_per_epoch)

    train_dataloader = DataLoader(
        cutmix_dataset if cutmix_dataset is not None else
        train_dataset, batch_size=batch_size,
        sampler=weighted_train_sampler, num_workers=4)

    test_dataset = DATASETS[dataset](
        split="val", seed=seed,
        image_size=(image_size, image_size),
        filter_mask_area=filter_mask_area)

    # RuntimeWarning divide by zero can happen, everything will work as it should,
    # but this means that some classes are not present in the validation dataset.
    class_weights = np.where(test_dataset.class_counts == 0, 0, 1.0 / test_dataset.class_counts)
    weights = [class_weights[label] for label in test_dataset.all_labels]

    val_sampler = WeightedRandomSampler(
        weights, replacement=True,
        num_samples=batch_size * iterations_per_epoch)

    val_dataloader = DataLoader(
        test_dataset, batch_size=batch_size,
        sampler=val_sampler, num_workers=4)

    model = ClassificationModel(
        train_dataset.num_classes,
        backbone=classifier_backbone
    ).cuda()

    # Check if the model is on CUDA
    if next(model.parameters()).is_cuda:
        print(f"Model is on CUDA and device is: {next(model.parameters()).device}")
    else:
        print("Model is NOT on CUDA")

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_validation_accuracy = 0
    best_model = None
    records = []

    for epoch in trange(num_epochs, desc="Training Classifier"):

        model.train()

        epoch_loss = torch.zeros(
            train_dataset.num_classes,
            dtype=torch.float32, device='cuda')
        epoch_accuracy = torch.zeros(
            train_dataset.num_classes,
            dtype=torch.float32, device='cuda')
        epoch_size = torch.zeros(
            train_dataset.num_classes,
            dtype=torch.float32, device='cuda')

        for image, label in train_dataloader:
            image, label = image.cuda(), label.cuda()

            logits = model(image)
            prediction = logits.argmax(dim=1)

            loss = F.cross_entropy(logits, label, reduction="none")
            if len(label.shape) > 1: label = label.argmax(dim=1)

            accuracy = (prediction == label).float()

            optim.zero_grad()
            loss.mean().backward()
            optim.step()

            with torch.no_grad():

                epoch_size.scatter_add_(0, label, torch.ones_like(loss))
                epoch_loss.scatter_add_(0, label, loss)
                epoch_accuracy.scatter_add_(0, label, accuracy)

        training_loss = epoch_loss / epoch_size.clamp(min=1)
        training_accuracy = epoch_accuracy / epoch_size.clamp(min=1)

        training_loss = training_loss.cpu().numpy()
        training_accuracy = training_accuracy.cpu().numpy()

        model.eval()

        epoch_loss = torch.zeros(
            train_dataset.num_classes,
            dtype=torch.float32, device='cuda')
        epoch_accuracy = torch.zeros(
            train_dataset.num_classes,
            dtype=torch.float32, device='cuda')
        epoch_size = torch.zeros(
            train_dataset.num_classes,
            dtype=torch.float32, device='cuda')

        for image, label in val_dataloader:
            image, label = image.cuda(), label.cuda()

            logits = model(image)
            prediction = logits.argmax(dim=1)

            loss = F.cross_entropy(logits, label, reduction="none")
            accuracy = (prediction == label).float()

            with torch.no_grad():
                epoch_size.scatter_add_(0, label, torch.ones_like(loss))
                epoch_loss.scatter_add_(0, label, loss)
                epoch_accuracy.scatter_add_(0, label, accuracy)

        validation_loss = epoch_loss / epoch_size.clamp(min=1)
        validation_accuracy = epoch_accuracy / epoch_size.clamp(min=1)

        validation_loss = validation_loss.cpu().numpy()
        validation_accuracy = validation_accuracy.cpu().numpy()  # it is necessary to not only save the mean

        # Check if the current epoch has the best validation accuracy
        if validation_accuracy.mean() > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy.mean()
            print(f"epoch: {epoch} | new best val acc: {best_validation_accuracy}")
            best_model = model.state_dict()

        records.append(dict(
            seed=seed,
            examples_per_class=examples_per_class,
            epoch=epoch,
            value=training_loss.mean(),
            metric="Loss",
            split="Training"
        ))

        records.append(dict(
            seed=seed,
            examples_per_class=examples_per_class,
            epoch=epoch,
            value=validation_loss.mean(),
            metric="Loss",
            split="Validation"
        ))

        records.append(dict(
            seed=seed,
            examples_per_class=examples_per_class,
            epoch=epoch,
            value=training_accuracy.mean(),
            metric="Accuracy",
            split="Training"
        ))

        records.append(dict(
            seed=seed,
            examples_per_class=examples_per_class,
            epoch=epoch,
            value=validation_accuracy.mean(),
            metric="Accuracy",
            split="Validation"
        ))

        for i, name in enumerate(train_dataset.class_names):
            records.append(dict(
                seed=seed,
                examples_per_class=examples_per_class,
                epoch=epoch,
                value=training_loss[i],
                metric=f"Loss {name.title()}",
                split="Training"
            ))

            records.append(dict(
                seed=seed,
                examples_per_class=examples_per_class,
                epoch=epoch,
                value=validation_loss[i],
                metric=f"Loss {name.title()}",
                split="Validation"
            ))

            records.append(dict(
                seed=seed,
                examples_per_class=examples_per_class,
                epoch=epoch,
                value=training_accuracy[i],
                metric=f"Accuracy {name.title()}",
                split="Training"
            ))

            records.append(dict(
                seed=seed,
                examples_per_class=examples_per_class,
                epoch=epoch,
                value=validation_accuracy[i],
                metric=f"Accuracy {name.title()}",
                split="Validation"
            ))
    if save_model:
        modeldir = os.path.join(os.path.dirname(logdir), "models")
        os.makedirs(modeldir, exist_ok=True)
        model_path = os.path.join(modeldir, f"classifier_{dataset}_{seed}_{examples_per_class}")
        if num_synthetic > 0:
            model_path = model_path + f"_{strength}_{guidance_scale}"
        if use_synthetic_filter:
            model_path = model_path + "_filter"
        if use_llm_prompt:
            model_path = model_path + "_llm"
        if use_embedding_noise:
            model_path = model_path + "_noise"
        if "test_uncommon" in eval_on_test_set:
            model_path = model_path + "_uncommon"
        model_path = model_path + ".pth"
        torch.save(best_model, model_path)
    if len(eval_on_test_set) > 0:
        # Load the best model for evaluation
        model.load_state_dict(best_model)
        model.eval()

        for test_set in eval_on_test_set:
            print(f'Evaluating {test_set} dataset...')
            # Build the test dataset
            test_dataset = DATASETS[dataset](split=test_set, seed=seed, image_size=(image_size, image_size))
            test_dataloader = DataLoader(test_dataset)

            epoch_loss = torch.zeros(test_dataset.num_classes, dtype=torch.float32, device='cuda')
            epoch_accuracy = torch.zeros(test_dataset.num_classes, dtype=torch.float32, device='cuda')
            epoch_size = torch.zeros(test_dataset.num_classes, dtype=torch.float32, device='cuda')
            for image, label in test_dataloader:
                image, label = image.cuda(), label.cuda()
                logits = model(image)
                prediction = logits.argmax(dim=1)
                loss = F.cross_entropy(logits, label, reduction="none")
                accuracy = (prediction == label).float()
                with torch.no_grad():
                    epoch_size.scatter_add_(0, label, torch.ones_like(loss))
                    epoch_loss.scatter_add_(0, label, loss)
                    epoch_accuracy.scatter_add_(0, label, accuracy)

            test_loss = epoch_loss / epoch_size.clamp(min=1)
            test_accuracy = epoch_accuracy / epoch_size.clamp(min=1)
            test_loss = test_loss.cpu().numpy()
            test_accuracy = test_accuracy.cpu().numpy()
            print(f'{test_set} accuracy: {test_accuracy.mean()}')

            testset_record = [dict(value=test_loss.mean(), metric=f"Mean Loss"),
                              dict(value=test_accuracy.mean(), metric=f"Mean Accuracy")]
            for i, name in enumerate(test_dataset.class_names):
                testset_record.append(dict(value=test_loss[i], metric=f"Loss {name.title()}"))
                testset_record.append(dict(value=test_accuracy[i], metric=f"Accuracy {name.title()}"))

            testdir = os.path.join(os.path.dirname(logdir), "test")
            os.makedirs(testdir, exist_ok=True)
            test_path = os.path.join(testdir, f"{test_set}_results_{dataset}_{seed}_{examples_per_class}.csv")
            pd.DataFrame.from_records(testset_record).to_csv(test_path)
            print(f"{test_set} record saved to: {test_path}")

    return records


class ClassificationModel(nn.Module):

    def __init__(self, num_classes: int, backbone: str = "resnet50"):

        super(ClassificationModel, self).__init__()

        self.backbone = backbone
        self.image_processor = None

        if backbone == "resnet50":

            self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.out = nn.Linear(2048, num_classes)

        elif backbone == "deit":

            self.base_model = DeiTModel.from_pretrained(
                "facebook/deit-base-distilled-patch16-224")
            self.out = nn.Linear(768, num_classes)

    def forward(self, image):

        x = image

        if self.backbone == "resnet50":

            with torch.no_grad():

                x = self.base_model.conv1(x)
                x = self.base_model.bn1(x)
                x = self.base_model.relu(x)
                x = self.base_model.maxpool(x)

                x = self.base_model.layer1(x)
                x = self.base_model.layer2(x)
                x = self.base_model.layer3(x)
                x = self.base_model.layer4(x)

                x = self.base_model.avgpool(x)
                x = torch.flatten(x, 1)

        elif self.backbone == "deit":

            with torch.no_grad():

                x = self.base_model(x)[0][:, 0, :]

        return self.out(x)


if __name__ == "__main__":
    '''
    Step 4:
    This script executes the last steps of the pipeline (generating synthetic images and training the downstream classifier).
    To run this skript the fine-tuned embeddings are needed (execute step 1 and 2 to get tokens).
    The classifier will be a fine-tuned version of classifier-backbone (resnet50) trained on a combination of real
    and synthetic data.

    Example call in terminal:
    python train_classifier.py --dataset "custom_coco" --synthetic-dir "intermediates/coco_ext_test/synthetic_class_concepts" --logdir "intermediates/coco_ext_test/logs" --iterations-per-epoch 200 --num-epochs 50 --batch-size 32 --num-synthetic 5 --num-trials 1 --examples-per-class 8 --embed-path "intermediates/coco_ext_test/custom_coco-tokens/custom_coco-0-2.pt" --aug "textual-inversion" --strength 0.5 --guidance-scale 7.5 --mask 0 --inverted 0 --use-generated-prompts 0
    
    python train_classifier.py --dataset "custom_coco" --examples-per-class 2 --seed 0 --strength 0.7 --guidance-scale 15 --synthetic-probability 0.7 --use-embedding-noise 1 --use-generated-prompts 1 --prompt-path "prompts/custom_coco_llama.csv" --synthetic_filter "train" --method "DIAGen" --eval_on_test_set "test" --num-synthetic 10 --num-epochs 50 --iterations-per-epoch 200 --device 0

    '''

    parser = argparse.ArgumentParser("Few-Shot Baseline")

    parser.add_argument("--logdir", type=str, default=os.path.join(DEFAULT_DIR, "logs"))
    # Directory used for logging and results
    parser.add_argument("--model-path", type=str, default="CompVis/stable-diffusion-v1-4")
    # Path to the Diffusion Model

    parser.add_argument("--prompt", type=str, default="a photo of a {name}")
    # A Textual Inversion parameter:
    # Augmentations are generated conditioned on the prompt ({name} is replaced with the particular class pseudo word)

    parser.add_argument("--use-generated-prompts", type=int, default=False)
    # Determines if prompts of LLM are used or the prompt(s) from the --prompts argument in the command line

    parser.add_argument("--prompt-path", type=str, default="prompts/prompts.csv")

    parser.add_argument("--use-embedding-noise", type=int, default=False)
    # Determines if noisy embeddings are used

    parser.add_argument("--synthetic-probability", type=float, default=0.7)
    # Probability to pick an image from the synthetic dataset while training the downstream model
    parser.add_argument("--synthetic-dir", type=str, default=os.path.join(DEFAULT_DIR, "synthetics_seed_{seed}"))
    # Directory to save the generated synthetic images

    parser.add_argument("--image-size", type=int, default=256)
    # Define the desired image size to convert all images to: [`image_size`, `image_size`]
    parser.add_argument("--classifier-backbone", type=str,
                        default="resnet50", choices=["resnet50", "deit"])
    # The pre-trained model to use
    parser.add_argument("--iterations-per-epoch", type=int, default=200)
    # Define how many different batches the classifier is trained on to complete an epoch
    parser.add_argument("--num-epochs", type=int, default=50)
    # Define how many epochs the training is running
    parser.add_argument("--batch-size", type=int, default=16)
    # Define how many images (real or synthetic) are in one batch

    parser.add_argument("--num-synthetic", type=int, default=10)
    # Define how many synthetic images should be generated per class

    parser.add_argument("--seeds", nargs='+', type=int, default=[0, 1, 2])
    # Define how often the entire experiment should be run with different seeds
    # Replaced --num-trials with --seeds. To enable custom seed setting
    # parser.add_argument("--num-trials", type=int, default=8)

    parser.add_argument("--examples-per-class", nargs='+', type=int, default=[2, 4, 8])
    # Define how many different images per class from the train data are used as guiding image
    # in the image generating process

    parser.add_argument("--embed-path", type=str, default=DEFAULT_EMBED_PATH)
    # Path to the trained embeddings of the pseudo words

    parser.add_argument("--dataset", type=str, default="custom_coco",
                        choices=["coco", "custom_coco", "focus"])

    parser.add_argument("--aug", nargs="+", type=str, default=["textual-inversion"],
                        choices=["real-guidance", "textual-inversion",
                                 "multi-token-inversion"])
    # We only use Textual Inversion

    parser.add_argument("--strength", nargs="+", type=float, default=None)
    # A StableDiffusionImg2ImgPipeline and StableDiffusionInpaintPipeline Parameter:
    # strength (`float`, *optional*, defaults to 0.8):
    #   Indicates extent to transform the reference image. Must be between 0 and 1. Image is used as a
    #   starting point and more noise is added the higher the `strength`. The number of denoising steps depends
    #   on the amount of noise initially added. A value of 1 essentially ignores the reference image.
    parser.add_argument("--guidance-scale", nargs="+", type=float, default=None)
    # A StableDiffusionImg2ImgPipeline and StableDiffusionInpaintPipeline Parameter:
    # guidance_scale (`float`, *optional*, defaults to 7.5):
    #   A higher guidance scale value encourages the model to generate images closely linked to the text prompt
    #   at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
    parser.add_argument("--mask", nargs="+", type=int, default=[0], choices=[0, 1])
    # A StableDiffusionInpaintPipeline Parameter:
    # mask_image (`torch.FloatTensor`):
    #   `mask_image` is representing an image batch to mask `image`. White pixels in the mask are repainted
    #   while black pixels are preserved. Mask determines which pixels the model is allowed to change.
    parser.add_argument("--inverted", nargs="+", type=int, default=[0], choices=[0, 1])
    # A Textual Inversion Parameter:
    #   Allows to invert the mask
    parser.add_argument("--probs", nargs="+", type=float, default=None)

    parser.add_argument("--compose", type=str, default="parallel",
                        choices=["parallel", "sequential"])

    parser.add_argument("--erasure-ckpt-path", type=str, default=None)
    # A Textual Inversion Parameter:
    #   Allows to erasure model knowledge to prevent data leakage as described in the DA-Fusion paper
    parser.add_argument("--use-randaugment", action="store_true")
    # Whether to use RandAugment or normal augmentation (rotation and flip)
    # RandAugment: Practical automated data augmentation with a reduced search space <https://arxiv.org/abs/1909.13719>
    parser.add_argument("--use-cutmix", action="store_true")
    # Whether to use CutMix or not
    # CutMix is an augmentation strategy for image data. Instead of removing pixels as in Cutout,
    # CutMix replaces the removed regions with a patch from another image.

    parser.add_argument("--tokens-per-class", type=int, default=4)
    # A Textual Inversion Parameter
    #   Only used when --aug "multi-token-inversion" selected, we do not use it

    parser.add_argument("--synthetic_filter", type=str, default=None,
                        choices=["use", "train", None])
    # Use a classifier as filter to determine the presence of the labelled class in the synthetically
    # generated images. "Use" will use a saved filter, "train" will train a new one.

    parser.add_argument("--filter_mask_area", type=int, default=0)
    # Determines how much images per class to filter out by area size of largest bounding box for pseudo word generation
    # If no filtering at all, set to zero
    # 'Good' value is 50000 and everything in the range of 30000 - 70000 works pretty well

    parser.add_argument("--device", type=int, default=0)
    # On which GPU to run
    parser.add_argument("--save_model", type=bool, default=True)
    # Whether to save the best classifier model or not
    parser.add_argument("--eval_on_test_set", nargs="+", type=str, default=[])
    # On which datasets the best classifier model should be evaluated
    # Custom coco has 2 choices: "test" and "test_uncommon"
    parser.add_argument("--method", type=str, default="baseline")
    # String containing information about the current run, used as directory name.
    # We use it to tag different methods for our ablation study

    args = parser.parse_args()

    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError:
        rank, world_size = 0, 1

    # device_id = rank % torch.cuda.device_count()
    # torch.cuda.set_device(rank % torch.cuda.device_count())
    device_id = args.device
    torch.cuda.set_device(device_id)

    print(f'Initialized process {rank} / {world_size} on current device(gpu) {torch.cuda.current_device()}')

    options = product(args.seeds, args.examples_per_class)
    options = np.array(list(options))
    options = np.array_split(options, world_size)[rank]

    for seed, examples_per_class in options.tolist():
        all_trials = []
        hyperparameters = dict(
            examples_per_class=examples_per_class,
            seed=seed,
            dataset=args.dataset,
            num_epochs=args.num_epochs,
            iterations_per_epoch=args.iterations_per_epoch,
            batch_size=args.batch_size,
            model_path=args.model_path,
            synthetic_probability=args.synthetic_probability,
            num_synthetic=args.num_synthetic,
            prompt=args.prompt,
            tokens_per_class=args.tokens_per_class,
            aug=args.aug,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            mask=args.mask,
            inverted=args.inverted,
            probs=args.probs,
            compose=args.compose,
            use_randaugment=args.use_randaugment,
            use_cutmix=args.use_cutmix,
            erasure_ckpt_path=args.erasure_ckpt_path,
            image_size=args.image_size,
            classifier_backbone=args.classifier_backbone,
            synthetic_filter=args.synthetic_filter,
            filter_mask_area=args.filter_mask_area,
            use_llm_prompt=args.use_generated_prompts,
            prompt_path=args.prompt_path,
            save_model=args.save_model,
            eval_on_test_set=args.eval_on_test_set,
            use_embedding_noise=args.use_embedding_noise,
            method=args.method)

        log_dir = args.logdir.format(**hyperparameters)
        os.makedirs(log_dir, exist_ok=True)
        synthetic_dir = args.synthetic_dir.format(**hyperparameters)
        embed_path = args.embed_path
        if embed_path == DEFAULT_EMBED_PATH and args.use_embedding_noise:
            embed_path = DEFAULT_NOISE_EMBED_PATH
        embed_path = embed_path.format(**hyperparameters)
        print("Use embedings at:", embed_path)

        all_trials.extend(run_experiment(
            synthetic_dir=synthetic_dir,
            embed_path=embed_path,
            logdir=log_dir,
            **hyperparameters))

        path = f"results_{args.dataset}_{seed}_{examples_per_class}.csv"
        path = os.path.join(log_dir, path)

        pd.DataFrame.from_records(all_trials).to_csv(path)
        print(f"[rank {rank}] n={examples_per_class} saved to: {path}")
