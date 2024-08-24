from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation

from PIL import Image
from typing import Tuple, Dict
import os
import glob
import numpy as np
import torchvision.transforms as transforms
import torch
import warnings
import matplotlib.pyplot as plt
import csv

base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
CUSTOM_COCO_DIR = os.path.join(base_dir, 'custom_coco')  # plus relative location to dataset


class CustomCOCO(FewShotDataset):
    classes = ["bench", "bicycle", "book", "bottle", "bowl", "car", "cell_phone", "chair", "clock", "computer_mouse",
               "cup", "fork", "keyboard", "knife", "laptop", "motorcycle", "spoon", "potted_plant", "sports_ball",
               "tie", "traffic_light", "tv_remote", "wine_glass"]
    class_names = sorted(classes)
    num_classes: int = len(class_names)

    def __init__(self, *args, data_dir: str = CUSTOM_COCO_DIR,
                 split: str = "train", seed: int = 0,
                 examples_per_class: int = None,
                 generative_aug: GenerativeAugmentation = None,
                 synthetic_probability: float = 0.5,
                 use_randaugment: bool = False,
                 image_size: Tuple[int] = (256, 256),
                 filter_mask_area: int = 0,  # Not used, but needs to change call of COCODataset to be removed
                 use_manual_list: bool = False,  # Not used
                 **kwargs):

        super(CustomCOCO, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability,
            generative_aug=generative_aug, **kwargs)

        # Create a dictionary to store class names and corresponding image paths
        # glob.glob is used to retrieve all files with a ".jpg" extension in these directories.
        self.image_paths = {class_name: [] for class_name in self.class_names}
        for class_name in self.class_names:
            if split == "test":
                class_dir_path = os.path.join(data_dir, 'test', class_name, '*.jpg')
            elif split == "test_uncommon":
                class_dir_path = os.path.join(data_dir, 'test_uncommon_context', class_name, '*.png')
            elif split == "train" or split == "val":
                class_dir_path = os.path.join(data_dir, 'train-val', class_name, '*.jpg')
            else:
                warnings.warn(f"Unknown split value: {split}. Using default train-val.", UserWarning)
                class_dir_path = os.path.join(data_dir, 'train-val', class_name, '*.jpg')
            class_image_paths = glob.glob(class_dir_path)
            self.image_paths[class_name].extend(class_image_paths)

        rng = np.random.default_rng(seed)
        # Generate random permutations of indices
        class_ids = {class_name: rng.permutation(len(self.image_paths[class_name]))
                     for class_name in self.class_names}

        class_ids_train, class_ids_val, class_ids_test = {}, {}, {}
        if split == "test" or split == "test_uncommon":
            class_ids_test = class_ids
        else:
            max_size_trainset = 8
            min_size_valset = 8
            # Split the shuffled indices into training and validation sets
            for class_name in self.class_names:
                if len(class_ids[class_name]) > max_size_trainset:
                    # First max_size_trainset images go to the training set
                    class_ids_train[class_name] = class_ids[class_name][:max_size_trainset]
                    # The rest go to the validation set
                    class_ids_val[class_name] = class_ids[class_name][max_size_trainset:]
                    if len(class_ids_val[class_name]) < min_size_valset:
                        warnings.warn(
                            f"Only {len(class_ids_val[class_name])} images in validation split for class {class_name}!",
                            UserWarning)
                else:
                    warnings.warn(f"Only {len(class_ids[class_name])} images in train-val for class {class_name}!",
                                  UserWarning)
                    # Move all images to training since there are less than max_size_trainset images in the dir
                    class_ids_train[class_name] = class_ids[class_name]

        # Select the training, validation or test indices based on the provided split parameter
        selected_class_ids = \
            {"train": class_ids_train, "val": class_ids_val, "test": class_ids_test, "test_uncommon": class_ids_test}[
                split]

        # Limits the number of examples per class
        if examples_per_class is not None and split == "train":
            for class_name in self.class_names:
                selected_class_ids[class_name] = selected_class_ids[class_name][:examples_per_class]

        # Checks for data imbalance
        critical_threshold = 5  # Warning if classes have fewer examples than critical_threshold

        if examples_per_class is not None and examples_per_class < critical_threshold:
            critical_threshold = examples_per_class
        critical_classes = [class_name for class_name in self.class_names
                            if len(selected_class_ids[class_name]) < critical_threshold]
        if critical_classes:
            warnings.warn(f"Warning for classes: {critical_classes} - fewer than "
                          f"{critical_threshold} examples for {split} split.")

        # Create the final list of images and labels for the chosen split
        self.all_images = [self.image_paths[class_name][idx] for class_name in self.class_names
                           for idx in selected_class_ids[class_name]]
        self.all_labels = [self.class_names.index(class_name) for class_name in self.class_names
                           for _ in selected_class_ids[class_name]]

        # Enumeration of the occurrences of each class in the data set
        self.class_counts = np.bincount(self.all_labels)

        # Writing image paths of training data to CSV
        out_dir_1 = "source_images"
        out_dir = "source_images/custom_coco"
        if not os.path.exists(out_dir_1):
            os.makedirs(out_dir_1)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        elif not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_path = os.path.join(out_dir, f"img_paths_{seed}_{examples_per_class}_{split}.csv")

        # Creating the CSV file and writing the paths to it
        with open(out_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for paths in self.all_images:
                row = [paths]
                writer.writerow(row)

        print(f"Wrote images paths of custom_coco {split}-split to csv: {out_path}")

        if use_randaugment:
            train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Lambda(lambda x: x.expand(3, *image_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.7, 1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15.0),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])

        val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        self.transform = \
            {"train": train_transform, "val": val_transform, "test": val_transform, "test_uncommon": val_transform}[
                split]

    def __len__(self):

        return len(self.all_images)

    def get_image_by_idx(self, idx: int) -> torch.Tensor:
        # return Image.open("/data/vilab05/CustomDatasets/Common_Objects/train-val/bench/photo_5267111192328001845_y.jpg").convert('RGB')
        return Image.open(self.all_images[idx]).convert('RGB')

    def get_label_by_idx(self, idx: int) -> torch.Tensor:

        return self.all_labels[idx]

    def get_metadata_by_idx(self, idx: int) -> Dict:
        # returns the class name of the image
        return dict(name=self.class_names[self.all_labels[idx]])

    def visualize_by_idx(self, idx: int):
        # Visualize the image specified by index from the dataset
        image_tensor, image_label = self.__getitem__(idx)

        # Ensure that the tensor is in the range [0, 1] for proper visualization
        image_tensor = (image_tensor + 1) / 2

        # Display the image using matplotlib
        plt.imshow(image_tensor.numpy().transpose(1, 2, 0))  # Transpose the dimensions for proper display
        plt.axis('off')
        plt.title(f'Label {image_label}, {self.class_names[image_label]}')
        plt.show()


if __name__ == "__main__":
    dataset = CustomCOCO(split="test", examples_per_class=4, seed=2)
    print('Dataset class counts:', dataset.class_counts)
    idx = 0
    dataset.visualize_by_idx(idx)
