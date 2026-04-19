# Pill and Anomaly Detection Training Model

## Overview
This repo contains all the tools to train a Faster R-CNN object detection model to detect pills and pill-related anomalies. The model is trained using semi-supervised learning with a self-learning approach thanks to the usage of pseudo-labels.

## How to Use the Repo

### General Setup
This codebase was made with Python 3.12.8. Any Python 3.12 version should be compatible. Other Python 3 versions may work too.

Clone the repo and run this command in a terminal to get all the required libraries:
`pip install -r requirements.txt`

### Basic Testing
The pre-trained model named `2026-04-18_15-12-48_best_frcnn.pt` can be tested by running `test.py`.
The `test.py` can be modified to test the model on any image file. Simply paste the image location as a parameter (`image_path`) for the function `single_test(device=None, model_path="faster_rcnn_pills.pt", image_path="test_pill.jpg")`. There are example tests that can be modified on line 80.

### Training the Model
To train the model, open the `train_with_pseudo_labels.py` file and run the function `start_training(load_custom, model_path=None, labeled_dataset_path=None, unlabel_dataset_path=None, new_name=None, not_semi=True)`.
Here are details on what each parameter represents:
- `load_custom`: A boolean that determines if the model you wish to train will be trained from the base COCO-trained model or based on an existing model. If TRUE, then the model specified in `model_path` will be retrained. If false, then `model_path` should be `None` and the basic Faster R-CNN model found on PyTorch will be trained.
- `model_path`: A string path to where the custom model is loaded.
- `labeled_dataset_path`: This training requires a labeled dataset. This is a string path to where the full dataset folder is located. The folder structure should contain a `train` and `valid` subfolder, and within those folders should be an `images` subfolder and an `_annotations.coco.json` file holding all the annotations.
- `unlabel_dataset_path`: This dataset is only required if you wish to use the semi-supervised/self-learning approach. Similar to the `labeled_dataset_path` parameter, but for the unlabeled dataset. There should not be any need for `_annotations.coco.json` files.
- `new_name`: This is automatically set in the code as the date and time the `train_with_pseudo_labels.py` is run, plus `_best_frcnn.pt`. This is the string name of the output model.
- `not_semi`: A boolean value that, if set to TRUE, will train the model with the supervised approach using only labeled data. If FALSE, then the model will be trained on the semi-supervised/self-learning approach.

The training process derives classes and functions from `faster_rcnn.py` (where the model creation code is) and `dataset.py` (for loading and parsing datasets).

The output of running the training code should be a newly trained model and output metrics represented by graphs showcasing the loss, F1-scores, and mAP scores for every epoch.

## Datasets Used
Listed below are the datasets that were used to train the current version of the model:
- Labeled dataset: https://universe.roboflow.com/cocotoyolo-rg00w/labeled_full_pill
- Unlabeled dataset: https://universe.roboflow.com/cocotoyolo-rg00w/unlabeled_full_pill
