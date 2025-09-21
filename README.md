# object_detection_dinov3: Lightweight head for object detection using DINOv3 as backbone

This repository provides a lightweight object detection head designed to run on top of Meta’s [DINOv3](https://github.com/facebookresearch/dinov3) backbone. The model combines a simple feature pyramid network (LightFPN) with an FCOS-style anchor-free detection head, enabling efficient detection on DINOv3 features without redundant computation. It has been trained using the [COCO dataset](https://cocodataset.org/).

This head is part of the [dinov3_ros](https://github.com/Raessan/dinov3_ros) project, where it enables real-time object detection in ROS 2 by reusing backbone features across multiple perception tasks.


## Table of Contents

1. [Installation](#installation)
2. [Model and loss function](#model_loss)
3. [Usage](#usage)
4. [Integration with dinov3_ros](#integration_dinov3_ros)
5. [Demo](#demo)
6. [License](#license)
7. [References](#references)


## Installation

We recommend using a fresh `conda` environment to keep dependencies isolated. DINOv3 requires Python 3.11, so we set that explicitly.

```
conda create -n object_detection_dinov3 python=3.11
conda activate object_detection_dinov3
git clone --recurse-submodules https://github.com/Raessan/object_detection_dinov3
cd object_detection_dinov3
pip install -e .
```

The only package that has to be installed separately is PyTorch, due to its dependence with the CUDA version. For example:

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129 
```

Finally, we provide weights for the lightweight heads developed by us in the `weights` folder, but the DINOv3 backbone weights should be requested and obtained from their [repo](https://github.com/facebookresearch/dinov3). Its default placement is in `dinov3_weights` folder. The presented head has been trained using the `vits16plus` model from DINOv3 as a backbone.

## Model and loss function

This repository implements a lightweight FCOS-style object detection head that can be attached to the [DINOv3](https://github.com/facebookresearch/dinov3) backbone (or any ViT producing a single spatial feature map). The design has three main components:

### Model architecture

#### LightFPN

- Projects backbone features into a fixed channel size.

- Builds a 3-level feature pyramid (P3, P4, P5) using strided convolutions and top-down upsampling.

- Keeps the design lightweight while allowing multi-scale detection.

#### FCOS Head

- Anchor-free detection head with three parallel branches:

    1) Classification → per-class logits at each location.

    2) Regression → distance offsets (l, t, r, b) from the location center to box edges.

    3) Centerness → scalar score measuring how close a location is to the object center.

- Designed to be efficient and easy to optimize with focal loss.

#### Full Head (DinoFCOSHead)

- Combines LightFPN and FCOSHead into a single module.

- Input: backbone feature map (B, C, H, W).

- Output: per-level predictions (cls, reg, ctr) for each feature map.

### Loss functions

Training uses a combination of three losses:

#### Classification loss: *Focal Loss*

- Down-weights easy negatives and focuses training on harder, misclassified examples.

#### Regression loss: *Generalized IoU (GIoU) Loss*

- Encourages predicted boxes to tightly cover ground truth, even when boxes do not overlap.

#### Centerness loss: Binary Cross-Entropy

- Ensures that locations closer to the object center are weighted higher during regression.

The total loss is a weighted sum:

$$
L_{total} = L_{cls} + \lambda_{rec} L_{rec} + \lambda_{ctr} L_{ctr}
$$

## Usage

There are three main folders and files that the user should use:

`config/config.py`: This file allows the user to configure the model, loss, training step or inference step. The parameters are described in the file.

`train/train_detector.ipynb`: Jupyter notebook for training the detector.

`inference/inference.py`: Script for running inference with a trained model on new images.

Additionally, the repository includes a `src` folder that contains the backend components: dataset utilities, backbone/head model definitions, and helper scripts. In particular:

- `common.py`: general-purpose functions that can be reused across different task-specific heads.
- `utils.py`: utilities tailored specifically for object detection (e.g., box processing, visualization).

The detector was trained for a total of 30 epochs: first for 15 epochs with a learning rate of 1e-4 using data augmentation, followed by 15 epochs with a reduced learning rate of 1e-5 without augmentation. The final weights have been placed in the `weights` folder.

Our main objective was not to surpass state-of-the-art models, but to train a head with solid results that enables collaboration and contributes to building a more refined [dinov3_ros](https://github.com/Raessan/dinov3_ros). This effort is particularly important because Meta has not released lightweight task-specific heads. For this reason, we welcome contributions — whether it’s improving this detection head, adding new features, or experimenting with alternative model architectures. Feel free to open an issue or submit a pull request! See the [Integration with dinov3_ros](#integration-dinov3_ros) section to be compliant with the [dinov3_ros](https://github.com/Raessan/dinov3_ros) project. 

## Integration with [dinov3_ros](https://github.com/Raessan/dinov3_ros)

This repository is designed to be easily integrated into [dinov3_ros](https://github.com/Raessan/dinov3_ros). To enable plug-and-play usage, the following files must be exported from the `src` folder to the `dinov3_toolkit/head_detection` folder in [dinov3_ros](https://github.com/Raessan/dinov3_ros):

- `model_head.py`: defines the detection head architecture.
- `utils.py`: task-specific utilites for object detection.
- `class_names.txt`: mapping from class indices to human-readable labels.

Additionally, we provide our chosen weights in `weights/model.pth`.

Any modification or extension of this repository should maintain these files and remain self-contained, so that the head can be directly plugged into [dinov3_ros](https://github.com/Raessan/dinov3_ros) without additional dependencies.


## Demo

<img src="assets/gif_object_detection.gif" height="800">

## License
- Code in this repo: Apache-2.0.
- DINOv3 submodule: licensed separately by Meta (see its LICENSE).
- We don't distribute DINO weights. Follow upstream instructions to obtain them.

## References

- [Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, Piotr Bojanowski (2025). Dinov3. *arXiv preprint arXiv:2508.10104.*](https://github.com/facebookresearch/dinov3)

- [Zhi Tian, Chunhua Shen, Hao Chen, Tong He (2019). FCOS: Fully Convolutional One-Stage Object Detection. *IEEE/CVF International Conference on Computer Vision (ICCV)*](https://ieeexplore.ieee.org/document/9010746)

- [Escarabajal, Rafael J. (2025). dinov3_ros](https://github.com/Raessan/dinov3_ros)