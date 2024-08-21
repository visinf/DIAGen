from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.coco_extension import COCOExtension
from semantic_aug.datasets.focus import FOCUS
#from semantic_aug.datasets.road_sign import RoadSignDataset
from models.filter_model import ClassificationFilterModel
from torch.utils.data import DataLoader, WeightedRandomSampler

import os
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

DATASETS = {
    "coco": COCODataset,
    "coco_extension": COCOExtension,
    #"road_sign": RoadSignDataset,
    "focus": FOCUS,
}


def train_filter(examples_per_class: int,
                 seed: int,
                 dataset: str,
                 image_size: int,
                 iterations_per_epoch: int = 200,
                 max_epochs: int = 50,
                 batch_size: int = 32,
                 model_dir: str = "models",
                 lr: float = 1e-4,
                 weight_decay: float = 1e-2,
                 use_randaugment: bool = True,
                 early_stopping_threshold: int = 6,
                 optimize_temperature: bool = True,
                 temp_optimizer_lr: float = 1e-2,
                 temp_optimizer_iterations: int = 500):
    """
    Trains a classifier on the training data using a weighted sampler to address imbalances in class distribution
    and saves the model version with the best validation loss.
    This saved model is intended for later use in filtering synthetic images.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset = DATASETS[dataset](
        split="train",
        examples_per_class=examples_per_class,
        synthetic_probability=0,
        use_randaugment=use_randaugment,
        seed=seed,
        image_size=(image_size, image_size)
    )

    # Calculate class weights based on the inverse of class frequencies. Assign weight to each sample in the dataset
    # based on the class distribution, so that each class has an equal contribution to the overall loss.
    # If class_count is 0 set the corresponding entry in class_weights to 0 too.
    class_weights = np.where(train_dataset.class_counts == 0, 0, 1.0 / train_dataset.class_counts)
    weights = [class_weights[label] for label in train_dataset.all_labels]

    weighted_train_sampler = WeightedRandomSampler(
        weights, replacement=True,
        num_samples=batch_size * iterations_per_epoch)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=weighted_train_sampler, num_workers=4)

    val_dataset = DATASETS[dataset](
        split="val", seed=seed,
        image_size=(image_size, image_size))

    # RuntimeWarning divide by zero can happen, everything will work as it should,
    # but this means that some classes are not present in the validation dataset.
    class_weights = np.where(val_dataset.class_counts == 0, 0, 1.0 / val_dataset.class_counts)
    weights = [class_weights[label] for label in val_dataset.all_labels]

    weighted_val_sampler = WeightedRandomSampler(
        weights, replacement=True,
        num_samples=batch_size * iterations_per_epoch)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size,
        sampler=weighted_val_sampler, num_workers=4)

    filter_model = ClassificationFilterModel(
        train_dataset.num_classes
    ).cuda()

    optim = torch.optim.Adam(filter_model.parameters(), lr=lr, weight_decay=weight_decay)

    best_validation_loss = np.inf
    corresponding_validation_accuracy = 0
    best_filter_model = None
    no_improvement_counter = 0

    records = []

    progress_bar = tqdm(range(max_epochs), desc="Training Filter")
    for epoch in progress_bar:

        filter_model.train()

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

            logits = filter_model(image)
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

        training_loss = training_loss.cpu().numpy().mean()
        training_accuracy = training_accuracy.cpu().numpy().mean()

        filter_model.eval()

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

            logits = filter_model(image)
            prediction = logits.argmax(dim=1)

            loss = F.cross_entropy(logits, label, reduction="none")
            accuracy = (prediction == label).float()

            with torch.no_grad():
                epoch_size.scatter_add_(0, label, torch.ones_like(loss))
                epoch_loss.scatter_add_(0, label, loss)
                epoch_accuracy.scatter_add_(0, label, accuracy)

        validation_loss = epoch_loss / epoch_size.clamp(min=1)
        validation_accuracy = epoch_accuracy / epoch_size.clamp(min=1)

        validation_loss = validation_loss.cpu().numpy().mean()
        validation_accuracy = validation_accuracy.cpu().numpy().mean()

        progress_bar.set_postfix({'train_loss': training_loss,
                                  'val_loss': validation_loss,
                                  'val_accuracy': validation_accuracy})

        records.append(dict(
            seed=seed,
            examples_per_class=0,
            epoch=epoch,
            value=training_loss,
            metric="Loss",
            split="Training"
        ))

        records.append(dict(
            seed=seed,
            examples_per_class=0,
            epoch=epoch,
            value=validation_loss,
            metric="Loss",
            split="Validation"
        ))

        records.append(dict(
            seed=seed,
            examples_per_class=0,
            epoch=epoch,
            value=training_accuracy,
            metric="Accuracy",
            split="Training"
        ))

        records.append(dict(
            seed=seed,
            examples_per_class=0,
            epoch=epoch,
            value=validation_accuracy,
            metric="Accuracy",
            split="Validation"
        ))

        # Check if the current epoch has the best validation loss
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            corresponding_validation_accuracy = validation_accuracy
            best_filter_model = filter_model.state_dict()
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        # Check for early stopping
        if no_improvement_counter >= early_stopping_threshold:
            print(
                f"No improvement in validation accuracy for {early_stopping_threshold} epochs. Stopping training.")
            break

    # Safe the logs
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/train_filter_{seed}_{epoch + 1}x{iterations_per_epoch}.csv"
    pd.DataFrame.from_records(records).to_csv(log_path)

    # Load the best model
    filter_model.load_state_dict(best_filter_model)

    if optimize_temperature:
        """
        Copyright (c) 2017 Geoff Pleiss
        https://github.com/gpleiss/temperature_scaling
        Tune the temperature of the model using the validation set.
        We're going to set it to optimize NLL.
        """
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # Collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for image, label in val_dataloader:
                image = image.cuda()

                logits = filter_model(image)

                logits_list.append(logits)
                labels_list.append(label)

            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()

        # Next: Optimize the temperature w.r.t. NLL
        temp_optimizer = torch.optim.LBFGS(
            [filter_model.temperature], lr=temp_optimizer_lr, max_iter=temp_optimizer_iterations)

        with tqdm(total=temp_optimizer_iterations, desc='Calibrating Filter') as pbar:
            def temp_eval():
                temp_optimizer.zero_grad()
                loss = nll_criterion(filter_model.temperature_scale(logits), labels)
                loss.backward()
                pbar.update(1)  # Update the progress bar
                pbar.set_postfix({'nll_loss': f'{loss.item():.4f}'})
                return loss

            temp_optimizer.step(temp_eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(filter_model.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(filter_model.temperature_scale(logits), labels).item()
        print('Optimal Temperature T: %.4f' % filter_model.temperature.item())
        print(f'Negative Log Likelihood (NLL) Loss improved from {before_temperature_nll:.4f} -> {after_temperature_nll:.4f}')
        print(f'Expected Calibration Error (ECE) improved from {before_temperature_ece:.4f} -> {after_temperature_ece:.4f}')

    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/filter_{dataset}_{seed}_{examples_per_class}.pth"
    torch.save(filter_model.state_dict(), model_path)

    print(f"Model saved to {model_path} - Validation loss {best_validation_loss} - Validation accuracy "
          f"{corresponding_validation_accuracy} - Training results saved to {log_path}")


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


if __name__ == "__main__":

    train_filter(examples_per_class=8,
                 seed=0,
                 dataset="focus",
                 image_size=256,
                 iterations_per_epoch=200,
                 max_epochs=50,
                 weight_decay=1e-2,
                 use_randaugment=True,
                 early_stopping_threshold=10,
                 optimize_temperature=True,
                 temp_optimizer_lr=1e-2,
                 temp_optimizer_iterations=500)
