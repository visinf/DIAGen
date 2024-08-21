# DIAGen
DIAGen: Semantically Diverse Image Augmentation with Generative Models for Few-Shot Learning (GCPR 2024)

## Diverse Image Augmentation with Generative Models
Simple data augmentation techniques, such as rotations and flips, are widely used to enhance the generalization power of computer vision models. However, these techniques often fail to modify high-level semantic attributes of a class. To address this limitation, researchers have explored generative augmentation methods like the recently proposed [DA-Fusion](https://arxiv.org/abs/2302.07944). Despite some progress, the variations are still largely limited to textural changes, thus falling short on aspects like varied viewpoints, environment, weather conditions, or even class-level semantic attributes (e.g., variations in a dog’s breed). To overcome this challenge, we propose DIAGen, building upon DA-Fusion. First, we apply Gaussian noise to the embeddings of an object learned with Textual Inversion to diversify generations using a pre-trained diffusion model’s knowledge. Second, we exploit the general knowledge of a text-to-text generative model to guide the image generation of the diffusion model with varied class-specific prompts. Finally, We introduce a weighting mechanism to mitigate the impact of poorly generated samples. Experimental results across various datasets show that DIAGen not only enhances semantic diversity but also improves the performance of subsequent classifiers. The advantages of DIAGen over standard augmentations and the DA-Fusion baseline are particularly pronounced with out-of-distribution samples.

## Installation (DIAGen)

To set up the repository, first clone the GitHub Repository.

```bash
git clone git@github.com:visinf/DIAGen.git
cd DIAGen
```

Then create the `conda` environment from the .yml File:

```bash
conda env create -f environment.yml
conda activate diagen
```

## Datasets

We benchmark DIAGen on multiple few-shot image classification problems, including the MSCOCO, FOCUS and CustomCOCO dataset (the latter) contains handmade images of 23 MSCOCO classes). For the MSCOCO, we label images with the classes corresponding to the largest object in the image. As you see in the `semantic_aug/datasets/` directory, there are more datasets available.

Custom datasets can be evaluated by implementing subclasses of `semantic_aug/few_shot_dataset.py`.

## Setting Up CustomCOCO dataset

The images of CustomCOCO dataset is part of this repo and located [here](https://github.com/visinf/DIAGen/blob/main/semantic_aug/datasets/custom_coco/)

## Setting Up FOCUS

An explanation on how to download the FOCUS dataset ([original repo](https://github.com/priyathamkat/focus.git)) can be found [here](https://umd.app.box.com/s/w7tvxer0wur7vtsoqcemfopgshn6zklv). After downloading and unzipping, execute our `semantic_aug/datasets/focus_create_split.py` to extract all the images into the needed directory structure and create a train, val and test split.

After that the FOCUS task is usable from `semantic_aug/datasets/focus.py`. `FOCUS_DIR` located [here](https://github.com/visinf/DIAGen/blob/main/semantic_aug/datasets/focus.py#L19) should be updated to point to the location of `focus` on your system.

## Setting Up MSCOCO

To set up MSCOCO, first download the [2017 Training Images](http://images.cocodataset.org/zips/train2017.zip), the [2017 Validation Images](http://images.cocodataset.org/zips/val2017.zip), and the [2017 Train/Val Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip). These files should be unzipped into the following directory structure.

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
TODO
```

