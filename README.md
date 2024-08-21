# DIAGen
DIAGen: Semantically Diverse Image Augmentation with Generative Models for Few-Shot Learning (GCPR 2024)

# Diverse Image Augmentation with Generative Models

Simple data augmentation techniques, such as rotations and flips, are widely used to enhance the generalization power of deep learning models. However, these methods often fail to introduce meaningful semantic diversity, such as variations in a dog's breed. Addressing this limitation, and building on a recently proposed method called [DA-Fusion](https://arxiv.org/abs/2302.07944), we explore how to use the general knowledge of generative models to increase semantic diversity in few-shot data. Our approach complements existing data augmentations by synthetically controlling image semantics particularly through prompts. Experimental results demonstrate that our method improves diversity and enhances classifier performance in downstream applications.

Our code builds upon [DA-Fusion](https://github.com/brandontrabucco/da-fusion).

## Installation (DIAGen)

To install the package, first create a `conda` environment.

```bash
conda create -n da-fusion python=3.7 pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6 -c pytorch
conda activate diagen
pip install diffusers["torch"] transformers pycocotools pandas matplotlib seaborn scipy
```

Then download and install the source code.

```bash
git clone git@github.com:visinf/DIAGen.git
pip install -e diagen
```

## Datasets

We benchmark DIAGen on multiple few-shot image classification problems, including the MSCOCO, FOCUS and CustomCOCO dataset (the latter) contains handmade images of 23 MSCOCO classes). For the MSCOCO, we label images with the classes corresponding to the largest object in the image. As you see in the `semantic_aug/datasets/` directory, there are more datasets available.

Custom datasets can be evaluated by implementing subclasses of `semantic_aug/few_shot_dataset.py`.

## Setting Up CustomCOCO dataset

The images of CustomCOCO dataset are located in the GitHub repo [Custom Datasets](https://github.com/Tobi-Tob/CustomDatasets.git) by Tobi-Tob and colleagues.

To access the images download them.

```bash
git clone git@github.com:Tobi-Tob/CustomDatasets.git
```

`COCO_EXTENSION_DIR` located [here](https://github.com/visinf/DIAGen/blob/main/semantic_aug/datasets/coco_extension.py#L15) should be updated to point to the location of `CustomDatasets/CommonObjects` on your system.
`ROAD_SIGN_DIR` located [here](https://github.com/visinf/DIAGen/blob/main/semantic_aug/datasets/road_sign.py#L15) should be updated to point to the location of `CustomDatasets/Road_Signs` on your system.

## Setting Up FOCUS

An explanation on how to download the FOCUS dataset ([original repo](https://github.com/priyathamkat/focus.git)) can be found [here](https://umd.app.box.com/s/w7tvxer0wur7vtsoqcemfopgshn6zklv). After downloading and unzipping, execute our `semantic_aug/datasets/focus_create_split.py` to extract all the images into the needed directory structure and create a train, val and test split.

After that the FOCUS task is usable from `semantic_aug/datasets/focus.py`. `FOCUS_DIR` located [here](https://github.com/visinf/DIAGen/blob/main/semantic_aug/datasets/focus.py#L15) should be updated to point to the location of `focus` on your system.

## Setting Up MSCOCO

To setup MSCOCO, first download the [2017 Training Images](http://images.cocodataset.org/zips/train2017.zip), the [2017 Validation Images](http://images.cocodataset.org/zips/val2017.zip), and the [2017 Train/Val Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip). These files should be unzipped into the following directory structure.

```
coco2017/
    train2017/
    val2017/
    annotations/
```

`COCO_DIR` located [here](https://github.com/visinf/DIAGen/blob/main/semantic_aug/datasets/coco.py#L17) should be updated to point to the location of `coco2017` on your system.

## Pipeline

The DIAGen pipeline consists of two major components. In the first step textual inversion (https://arxiv.org/abs/2208.01618) is used to extract class representations in the embedding space of Stable Diffusion. In order to do this the `fine_tune.py` needs to be performed.

Before going to the next component the fine-tuned vectors need to be aggregated to class embeddings. This is done by performing `aggregate_embeddings.py`. In this step, our first contribution can be used by the parameter `--augment-embeddings`. It works as described in the paper.

In the second component, DIAGen generates the synthetic data. For our second contribution, the class-specific text-prompts used by Stable Diffusion, the `generate_prompts.py` script needs to be executed. If you want to generate the text-prompts via a GPT model, it is necessary to add an api key in a .env file [here in the project directory](https://github.com/visinf/DIAGen).

To use our third contribution, the weighting mechanism for the synthetic images, execute `train_filter.py`.

The actual image generation process is combined with the training of the downstream classifier in `train_classifier.py`. This script accepts a number of arguments that control how the classifier is trained:

```bash
python train_classifier.py --dataset "focus" --examples-per-class 4 \
--strength 0.7 --guidance-scale 15 --synthetic-probability 0.7 --use-embedding-noise 1 \
--use-generated-prompts 1 --prompt-path "prompts/focus/prompts_gpt4.csv" \
--synthetic_filter "train" --method "noise_llm_filter" --eval_on_test_set "test" \
--num-synthetic 10 --num-epochs 50 --iterations-per-epoch 200 --device 0
```

This example will train a classifier on the FOCUS task, with 4 images per class, using the prompts located at `prompts/focus/prompts_gpt4.csv`. Slurm scripts that reproduce the paper are located in `scripts/textual_inversion`. Results are logged to `.csv` files based on the script argument `--logdir`. More detailed explanation of each argument can be found in the code

## Citation

If you find our method helpful, consider citing our preprint!

```
OUR CITATION
```

