from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict

import numpy as np
import torchvision.transforms as transforms
import torch
import os
import csv
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from PIL import Image
from collections import defaultdict

COCO_DIR = r"/data/vilab06/coco2017"  # put ur own path here

TRAIN_IMAGE_DIR = os.path.join(COCO_DIR, "train2017")
VAL_IMAGE_DIR = os.path.join(COCO_DIR, "val2017")

DEFAULT_TRAIN_INSTANCES = os.path.join(
    COCO_DIR, "annotations/instances_train2017.json")
DEFAULT_VAL_INSTANCES = os.path.join(
    COCO_DIR, "annotations/instances_val2017.json")


class COCODataset(FewShotDataset):

    """
0	person
1	bicycle
2	car
3	motorcycle
4	airplane
5	bus
6	train
7	truck
8	boat
9	trafficlight
10	firehydrant
11	stopsign
12	parkingmeter
13	bench
14	bird
15	cat
16	dog
17	horse
18	sheep
19	cow
20	elephant
21	bear
22	zebra
23	giraffe
24	backpack
25	umbrella
26	handbag
27	tie
28	suitcase
29	frisbee
30	skis
31	snowboard
32	sportsball
33	kite
34	baseballbat
35	baseballglove
36	skateboard
37	surfboard
38	tennisracket
39	bottle
40	wineglass
41	cup
42	fork
43	knife
44	spoon
45	bowl
46	banana
47	apple
48	sandwich
49	orange
50	broccoli
51	carrot
52	hotdog
53	pizza
54	donut
55	cake
56	chair
57	couch
58	pottedplant
59	bed
60	diningtable
61	toilet
62	tv
63	laptop
64	mouse
65	remote
66	keyboard
67	cellphone
68	microwave
69	oven
70	toaster
71	sink
72	refrigerator
73	book
74	clock
75	vase
76	scissors
77	teddybear
78	hairdrier
79	toothbrush
    """

    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                   'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                   'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                   'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                   'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                   'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                   'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                   'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                   'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                   'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    num_classes: int = len(class_names)

    def __init__(self, *args, split: str = "train", seed: int = 0,
                 train_image_dir: str = TRAIN_IMAGE_DIR,
                 val_image_dir: str = VAL_IMAGE_DIR,
                 train_instances_file: str = DEFAULT_TRAIN_INSTANCES,
                 val_instances_file: str = DEFAULT_VAL_INSTANCES,
                 examples_per_class: int = None,
                 generative_aug: GenerativeAugmentation = None,
                 synthetic_probability: float = 0.5,
                 use_randaugment: bool = False,
                 image_size: Tuple[int] = (256, 256),
                 filter_mask_area: int = 0,
                 use_manual_list: bool = False,
                 use_llm_prompt: bool = False,
                 prompt_path: str = None, **kwargs):

        super(COCODataset, self).__init__(
            *args, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability, 
            generative_aug=generative_aug,
            use_llm_prompt=use_llm_prompt,
            prompt_path=prompt_path, **kwargs)

        image_dir = {"train": train_image_dir, "val": val_image_dir}[split]
        instances_file = {"train": train_instances_file, "val": val_instances_file}[split]

        class_to_images = defaultdict(list)
        class_to_annotations = defaultdict(list)

        self.cocoapi = COCO(instances_file)
        for image_id, x in self.cocoapi.imgs.items():

            annotations = self.cocoapi.imgToAnns[image_id]
            if len(annotations) == 0: continue

            maximal_ann = max(annotations, key=lambda x: x["area"])
            class_name = self.cocoapi.cats[maximal_ann["category_id"]]["name"]
            ''' Code from us to avoid images where the class covers a small area
            if maximal_ann["area"] <= filter_mask_area:
                threshold = filter_mask_area
                exception_list = ["skis", "sports ball", "baseball bat", "fork", "knife", "spoon", "hair drier",
                                  "toothbrush"]
                if class_name in exception_list:
                    threshold = threshold / 4
                if maximal_ann["area"] <= threshold:
                    continue
            '''

            class_to_images[class_name].append(
                os.path.join(image_dir, x["file_name"]))
            class_to_annotations[class_name].append(maximal_ann)

        '''
        if use_manual_list:
            for category, img_list in manually_selected_imgs.items():
                class_to_images[category] = [os.path.join(image_dir, img) for img in img_list]
        '''

        rng = np.random.default_rng(seed)
        class_to_ids = {key: rng.permutation(
            len(class_to_images[key])) for key in self.class_names}

        if examples_per_class is not None:
            class_to_ids = {key: ids[:examples_per_class]
                            for key, ids in class_to_ids.items()}

        self.class_to_images = {
            key: [class_to_images[key][i] for i in ids]
            for key, ids in class_to_ids.items()}

        self.class_to_annotations = {
            key: [class_to_annotations[key][i] for i in ids]
            for key, ids in class_to_ids.items()}

        self.all_images = sum([
            self.class_to_images[key]
            for key in self.class_names], [])

        self.all_annotations = sum([
            self.class_to_annotations[key]
            for key in self.class_names], [])

        self.all_labels = [i for i, key in enumerate(
            self.class_names) for _ in self.class_to_images[key]]

        # Enumeration of the occurrences of each class in the data set
        self.class_counts = np.bincount(self.all_labels)

        # Writing image paths of training data to CSV
        out_dir_1 = "source_images"
        out_dir = "source_images/coco"
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

        print(f"Wrote images paths of coco {split}-split to csv: {out_path}")

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
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15.0),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Lambda(lambda x: x.expand(3, *image_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])

        val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        self.transform = \
            {"train": train_transform, "val": val_transform, "test": val_transform, "test_uncommon": val_transform}[
                split]

    def __len__(self):

        return len(self.all_images)

    def get_image_by_idx(self, idx: int) -> torch.Tensor:

        return Image.open(self.all_images[idx]).convert('RGB')

    def get_label_by_idx(self, idx: int) -> torch.Tensor:

        return self.all_labels[idx]

    def get_metadata_by_idx(self, idx: int) -> Dict:

        annotation = self.all_annotations[idx]

        return dict(name=self.class_names[self.all_labels[idx]],
                    mask=self.cocoapi.annToMask(annotation),
                    **annotation)

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
