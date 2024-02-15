# ca-redist
Code for Cross-attention Control for better Object assignment in ControlNet

# Thanks
This code is based on https://github.com/lllyasviel/ControlNet-v1-1-nightly


# Repo Structure:
* `experiments/` contains the results of the experiments. Each subfolder of this folder is one experiment (one setting), each containing the outputs for different datasets of a certain generation setting.
* `pretrained/` contains pre-trained models, such as the fine-tuned segmentation-based ControlNet hint blocks.
* `models/` should contain the pre-trained ControlNet v1.1 checkpoints, downloaded from HuggingFace Hub.
* `generate_controlnet_pww.py` is the main file that is used for generating images for a directory with a certain setting.
* `controlnet_pww.py` contains the implementations of various cross-attention control methods used here.
* `evaluation.py` contains the code for evaluation.
* `evaldata/` contains our SimpleScenes data and more.

The rest of the code is either (modified) code from the ControlNet v1.1 repo or some helper libraries and notebooks.


# How to use:
General generation workflow proceeds as follows:
1. create a folder in `experiments/` and create an `args.json` file with the specifications of the generation method and cross-attention control method.
2. run `generate_controlnet_pww.py` while specifying the experiment folder (which must contain `args.json`), as well as the datasets.


To rerun settings from the paper, pick the right folder in `experiments/` and run it using `generate_controlnet_pww.py` with the right set of datasets.

To run generation on COCO 2017, first download COCO 2017 with panoptic annotations, put it in a folder, and run `generate_controlnet_pww_coco.py`, pointing it to the COCO folder.