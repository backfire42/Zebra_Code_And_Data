# Zebrafish identification with deep CNN and ViT architectures using a rolling training window [\[paper\]](https://www.nature.com/articles/s41598-025-86351-x)
**This repository provides the code and data for our *Scientific Reports* paper "Zebrafish identification with deep CNN and ViT architectures using a rolling training window".**
### [Jason Puchalla](https://phy.princeton.edu/people/jason-puchalla), [Aaron Serianni](https://aaronserianni.com/), Bo Deng
> **Abstract:** *Zebrafish are widely used in vertebrate studies, yet minimally invasive individual tracking and identification in the lab setting remain challenging due to complex and time-variable conditions. Advancements in machine learning, particularly neural networks, offer new possibilities for developing simple and robust identification protocols that adapt to changing conditions. We demonstrate a rolling window training technique suitable for use with open-source convolutional neural networks (CNN) and vision transformers (ViT) that shows promise in robustly identifying individual maturing zebrafish in groups over several weeks. The technique provides a high-fidelity method for monitoring the temporally evolving zebrafish classes, potentially significantly reducing the need for new training images in both CNN and ViT architectures. To understand the success of the CNN classifier and inform future real-time identification of zebrafish, we analyzed the impact of shape, pattern, and color by modifying the images of the training set and compared the test results with other prevalent machine learning models.*

## Table of Contents
| Links                                                   |
|---------------------------------------------------------|
| [Dataset Guide](#dataset-guide)                         |
| [Repository Guide](#repository-guide)                   |
| [Monte Carlo Simulation](#monte-carlo-simulation)       |
| [Usage](#usage)                                         |
| ├ [classifier_comparison.py](#classifier_comparisonpy)  |
| ├ [crop.py](#croppy)                                    |
| ├ [evaluate_model.py](#evaluate_modelpy)                |
| ├ [get_model_flops.py](#get_model_flopspy)              |
| ├ [make_arrays.py](#make_arrayspy)                      |
| ├ [pad.py](#padpy)                                      |
| ├ [rotate.py](#rotatepy)                                |
| ├ [split_data_numbers.py](#split_data_numberspy)        |
| ├ [split_data.py](#split_datapy)                        |
| ├ [test_model.py](#test_modelpy)                        |
| └ [train_model.py](#train_modelpy)                      |
| ├ [test_model_vit.py](#test_model_vitpy)                |
| └ [train_model_vit.py](#train_model_vitpy)              |
| [License](#license)                                     |

## Resources
Materials related to the paper and this reposority:  

| Link  | Description       |
|-------|-------------------|
|       | PDF of Paper      |
| Here! | Source Code       |
|       | Zebrafish Dataset |

## Dataset Guide
The five different datasets corresponding to each day are stored in [`data.zip`](data.zip). 

Each dataset contains images of the five fish, labeled correspondingly across the days. In addition, each day has its own train-validation-test split. Manually blurred images from Day 8 are also included.

Raw, unprocessed images will be made available in the Princeton Research Data Repository at [doi.org/10.34770/pz36-j044](https://www.doi.org/10.34770/pz36-j044).

## Repository Guide

| Path                                                          | Description                                     |
|---------------------------------------------------------------|-------------------------------------------------|
| Zebra_Code_And_Data                                           | Root folder.                                    |
| ├ [data.zip](data.zip)                                        | Pre-processed dataset.                          |
| ├ [release](code)                                             | Final source code.                              |
|  ├ [classifier_comparison.py](code/classifier_comparison.py)  | Train and evaluate models from scikit-learn.    |
|  ├ [crop.py](code/crop.py)                                    | Crop zebrafish within fish studio.              |
|  ├ [evaluate_model.py](code/evaluate_modelpy)                 | Evaluate a model on images.                     |
|  ├ [get_model_flops.py](code/get_model_flopspy)               | Estimate number of flops required to run model. |
|  ├ [make_arrays.py](code/make_arrays.py)                      | Generate arrays with image augmentation.        |
|  ├ [pad.py](code/pad.py)                                      | Pad images to ratio.                            |
|  ├ [rotate.py](code/rotate.py)                                | Rotate images.                                  |
|  ├ [split_data_numbers.py](code/split_data_numberspy)         | Split data with set number of images in train.  |
|  ├ [split_data.py](code/split_data.py)                        | Split data in train, validation, and test.      |
|  ├ [test_model.py](code/test_model.py)                        | Test a trained CNN model on images.             |
|  ├ [train_model.py](code/train_model.py)                      | Train a CNN model.                              |
|  ├ [test_model.py](code/test_model.py)                        | Test a trained ViT model on images.             |
|  └ [train_model.py](code/train_model.py)                      | Train a ViT model.                              |
|  └ [zebrafish.yml](code/zebrafish.yml)                        | YAML file to create conda environment.          |
| └ [ZebraMappingSim_v3.mlx](ZebraMappingSim_v3.mlx)            | Monte Carlo simulation code                     |

## Monte Carlo Simulation
Matlab MC code ([`ZebraMappingSim_v3.mlx`](ZebraMappingSim_v3.mlx)) to predict the number of images needed for a rolling window DL model for zebrafish classification.

This Monte Carlo simulation uses the average measured successful matching probability between two image data sets collected at different times where there is some feature changes in the classes but no change in the number of classes. The probability of successful cross-day matching of each class is treated as independent.

## Usage
Create a conda environment using `conda env create -f environment.yml python=3.10`.

Pathway to run code: `crop.py` → `rotate.py` → `pad.py` → `split_data.py` → `make_arrays.py` → `train_model.py` → `test_model.py`.

In the arguments, `scratch_dir` is the folder where model checkpoints, TensorBoard logs, and TFHub modules are stored. `data_dir` is the location of the target dataset day (e.g., `data/day8/`).

### [`crop.py`](code/crop.py)
```
usage: crop.py data_dir [-h] 
               [-t TOLERANCE] [-c CROP CROP CROP CROP]
               [--blur1 BLUR1][--median MEDIAN] [--blur2 BLUR2]  

positional arguments:
  data_dir              Data Directory

optional arguments:
  -h, --help            show this help message and exit
  -t TOLERANCE, --tolerance TOLERANCE
                        Tolerance for chroma keying (default: 15)
  -c CROP CROP CROP CROP, --crop CROP CROP CROP CROP
                        Number of pixels to crop border for left, bottom, right, and top sides respectively (default for 2018-8-21 dataset: 560 660 230 1230)
  --blur1 BLUR1         First Gaussian blur radius (default: 50)
  --median MEDIAN       Median filter size (default: 9)
  --blur2 BLUR2         Second Gaussian blur radius (default: 30)
```

### [`rotate.py`](code/rotate.py)
```
usage: rotate.py data_dir [-h] [-r ROTATE] [--ignore_ratio]

positional arguments:
  data_dir              Data Directory
 
optional arguments:
  -h, --help            show this help message and exit
  -r ROTATE, --rotate ROTATE
                        Number of degrees to rotate image clockwise (default: 90)
  --ignore_ratio        Rotate image even if width is less than height
```

### [`pad.py`](code/pad.py)
```
usage: pad.py data_dir [-h] [-r RATIO RATIO]

positional arguments:
  data_dir              Data Directory

optional arguments:
  -h, --help            show this help message and exit
  -r RATIO RATIO, --ratio RATIO RATIO
                        Aspect ratio to compare image to (default: 16.0 9.0)
```

### [`split_data.py`](code/split_data.py)
```
usage: split_data.py [-h] [-n NUM_CLASS] [--name_class NAME_CLASS]
                     [--train_size TRAIN_SIZE]
                     [--validation_size VALIDATION_SIZE]
                     [--test_size TEST_SIZE]
                     data_dir

positional arguments:
  data_dir              Dataset Directory

optional arguments:
  -h, --help            show this help message and exit
  -n NUM_CLASS, --num_class NUM_CLASS
                        Number of classes (default: 5)
  --name_class NAME_CLASS
                        Prefix for class (default: fish)
  --train_size TRAIN_SIZE
                        Fraction of files to include in train split (default:
                        0.7)
  --validation_size VALIDATION_SIZE
                        Fraction of files to include in validation split
                        (default: 0.15)
  --test_size TEST_SIZE
                        Fraction of files to include in test split (default:
                        0.15)
```

### [`split_data_number.py`](code/split_data_number.py)
```
usage: split_data_number.py [-h] [-n NUM_CLASS] [--name_class NAME_CLASS] [--train_num TRAIN_NUM]
                            [--validation_size VALIDATION_SIZE] [--test_size TEST_SIZE]
                            data_dir source_dir

positional arguments:
  data_dir              Target Dataset Directory
  source_dir            Source Directory

options:
  -h, --help            show this help message and exit
  -n NUM_CLASS, --num_class NUM_CLASS
                        Number of classes (default: 5)
  --name_class NAME_CLASS
                        Prefix for class (default: fish)
  --train_num TRAIN_NUM
                        Number of files to include in train split (default: 10)
  --validation_size VALIDATION_SIZE
                        Fraction of files to include in validation split (default: 0.15)
  --test_size TEST_SIZE
                        Fraction of files to include in test split (default: 0.15)
```

### [`make_arrays.py`](code/make_arrays.py)
```
usage: make_arrays.py scratch_dir data_dir [-h] 
                      [-b BATCH_SIZE] [-n NUM_TRAIN]
                      [-i IMAGE_SHAPE IMAGE_SHAPE]
                      [--rotation_range ROTATION_RANGE]
                      [--width_shift_range WIDTH_SHIFT_RANGE]
                      [--height_shift_range HEIGHT_SHIFT_RANGE]
                      [--brightness_range BRIGHTNESS_RANGE BRIGHTNESS_RANGE]
                      [--shear_range SHEAR_RANGE] [--zoom_range ZOOM_RANGE]
                      [--channel_shift_range CHANNEL_SHIFT_RANGE]
                      [--no_vertical_flip] [--no_horizontal_flip]

positional arguments:
  scratch_dir           Scratch Directory
  data_dir              Dataset Directory

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size during data generation (default: 50)
  -n NUM_TRAIN, --num_train NUM_TRAIN
                        Number of training images (default: 10000)
  -i IMAGE_SHAPE IMAGE_SHAPE, --image_shape IMAGE_SHAPE IMAGE_SHAPE
                        Desired data image dimensions (default: 180 320)
  --rotation_range ROTATION_RANGE
                        Image Augmentation - Degree range for random rotations (default: 360)
  --width_shift_range WIDTH_SHIFT_RANGE
                        Image Augmentation - Range for fraction of total image width for random horizontal shifts (default: 0.1)
  --height_shift_range HEIGHT_SHIFT_RANGE
                        Image Augmentation - Range for fraction of total image height for random vertical shifts (default: 0.1)
  --brightness_range BRIGHTNESS_RANGE BRIGHTNESS_RANGE
                        Image Augmentation - Range for random brightness shift values (default: 0.3 1.3)
  --shear_range SHEAR_RANGE
                        Image Augmentation - Range for random shear intesity values (Shear angle in counter-clockwise direction in degrees/360) (default: 0.05)
  --zoom_range ZOOM_RANGE
                        Image Augmentation - Range for random zoom (default: 0.1)
  --channel_shift_range CHANNEL_SHIFT_RANGE
                        Image Augmentation - Range for random channel shifts (default: 0.1)
  --no_vertical_flip    Image Augmentation - Disable random vertical flips
  --no_horizontal_flip  Image Augmentation - Disable random horizontal flips
```

### [`train_model.py`](code/train_model.py)
```
usage: train_model.py scratch_dir data_dir [-h] 
                      [--num_gpu NUM_GPU] [--num_class NUM_CLASS]
                      [--no_weights] [--train_grayscale] [--test_grayscale]
                      [-l] [--layer_cuttoff LAYER_CUTT`OFF] [-e EPOCHS]
                      [-b BATCH_SIZE] [--checkpoint_name CHECKPOINT_NAME]
                      [--patience PATIENCE] [--model_name MODEL_NAME]

positional arguments:
  scratch_dir           Scratch Directory
  data_dir              Dataset Directory

optional arguments:
  -h, --help            show this help message and exit
  --num_gpu NUM_GPU     Number of GPUs (default: 4)
  --num_class NUM_CLASS
                        Number of classes in dataset (default: 5)
  --no_weights          Load Inception model without weights
  --train_grayscale     Make train and validation data grayscale
  --test_grayscale      Make test data grayscale
  -l, --labels          Display class labels on Tensorboard images
  --layer_cuttoff LAYER_CUTTOFF
                        InceptionV3 layer to end model at (example: mixed7)
  -e EPOCHS, --epochs EPOCHS
                        Maximum number of epochs (default: 200)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size during training (default: 50)
  --checkpoint_name CHECKPOINT_NAME
                        Name of checkpoint file where the best trained weights are stored (default: weights.hdf5)
  --patience PATIENCE   Early stopping patience in number of epochs (default: 10)
  --model_name MODEL_NAME
                        Name of json file where the model is stored (default: model.json)
```

### [`test_model.py`](code/test_model.py)
```
usage: test_model.py scratch_dir data_dir [-h] 
                     [--num_gpu NUM_GPU] [-b BATCH_SIZE]
                     [-i IMAGE_SIZE IMAGE_SIZE]
                     [--checkpoint_name CHECKPOINT_NAME]
                     [--model_name MODEL_NAME]

positional arguments:
  scratch_dir           Scratch Directory
  data_dir              Data Directory

optional arguments:
  -h, --help            show this help message and exit
  --num_gpu NUM_GPU     Number of GPUs. Must match trained model parameters (default: 4)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size during training (default: 50)
  -i IMAGE_SIZE IMAGE_SIZE, --image_size IMAGE_SIZE IMAGE_SIZE
                        Data image dimensions. Must match model input image dimension (default: 180 320)
  --checkpoint_name CHECKPOINT_NAME
                        Name of checkpoint file where the best trained weights are stored (default: weights.hdf5)
  --model_name MODEL_NAME
                        Name of json file where the model is stored (default: model.json)
```

### [`train_model_vit.py`](code/train_model_vit.py)
```
usage: train_model_vit.py [-h] [--num_class NUM_CLASS] [--no_weights] [--train_grayscale]
                          [--test_grayscale] [-l] [--layer_cuttoff LAYER_CUTTOFF] [-e EPOCHS]
                          [-b BATCH_SIZE] [--checkpoint_name CHECKPOINT_NAME] [--patience PATIENCE]
                          [--model_name MODEL_NAME]
                          scratch_dir data_dir

positional arguments:
  scratch_dir           Scratch Directory
  data_dir              Dataset Directory

options:
  -h, --help            show this help message and exit
  --num_class NUM_CLASS
                        Number of classes in dataset (default: 5
  --no_weights          Load Inception model without weights
  --train_grayscale     Make train and validation data grayscale
  --test_grayscale      Make test data grayscale
  -l, --labels          Display class labels on Tensorboard images
  --layer_cuttoff LAYER_CUTTOFF
                        InceptionV3 layer to end model at (example: mixed7)
  -e EPOCHS, --epochs EPOCHS
                        Maximum number of epochs (default: 200)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size during training (default: 50)
  --checkpoint_name CHECKPOINT_NAME
                        Name of checkpoint file where the best trained weights are stored (default:
                        weights.hdf5)
  --patience PATIENCE   Early stopping patience in number of epochs (default: 10)
  --model_name MODEL_NAME
                        Name of json file where the model is stored (default: model.json)
```

### [`test_model_vit.py`](code/test_model_vit.py)
```
usage: test_model_vit.py [-h] [-b BATCH_SIZE] [-i IMAGE_SIZE IMAGE_SIZE]
                         [--checkpoint_name CHECKPOINT_NAME] [--model_name MODEL_NAME]
                         scratch_dir data_dir

positional arguments:
  scratch_dir           Scratch Directory
  data_dir              Data Directory

options:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size during training (default: 50)
  -i IMAGE_SIZE IMAGE_SIZE, --image_size IMAGE_SIZE IMAGE_SIZE
                        Data image dimensions. Must match model input image dimension (default: 180 320)
  --checkpoint_name CHECKPOINT_NAME
                        Name of checkpoint file where the best trained weights are stored (default:)
                        weights.hdf5
  --model_name MODEL_NAME
                        Name of json file where the model is stored (default: model.json)
```

### [`evaluate_model.py`](code/evaluate_model.py)
```
usage: evaluate_model.py [-h] [--num_gpu NUM_GPU] [--num_class NUM_CLASS] [--name_class NAME_CLASS]
                         [-b BATCH_SIZE] [-i IMAGE_SIZE IMAGE_SIZE] [--checkpoint_name CHECKPOINT_NAME]
                         [--model_name MODEL_NAME]
                         scratch_dir data_dir

positional arguments:
  scratch_dir           Scratch Directory
  data_dir              Data Directory

options:
  -h, --help            show this help message and exit
  --num_gpu NUM_GPU     Number of GPUs. Must match trained model parameters (default: 4)
  --num_class NUM_CLASS
                        Number of classes in dataset (default: 5)
  --name_class NAME_CLASS
                        Number of classes in dataset (default: fish)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size during training (default: 50)
  -i IMAGE_SIZE IMAGE_SIZE, --image_size IMAGE_SIZE IMAGE_SIZE
                        Data image dimensions. Must match model input image dimension (default: 180 320)
  --checkpoint_name CHECKPOINT_NAME
                        Name of checkpoint file where the best trained weights are stored (default:
                        weights.hdf5)
  --model_name MODEL_NAME
                        Name of json file where the model is stored (default: model.json)
```

### [`get_model_flops.py`](code/get_model_flops.py)
```
usage: get_model_flops.py [-h] [--checkpoint_name CHECKPOINT_NAME] [-i IMAGE_SIZE IMAGE_SIZE]
                          [--model_name MODEL_NAME]
                          scratch_dir

positional arguments:
  scratch_dir           Scratch Directory

options:
  -h, --help            show this help message and exit
  --checkpoint_name CHECKPOINT_NAME
                        Name of checkpoint file where the best trained weights are stored (default:
                        weights.hdf5)
  -i IMAGE_SIZE IMAGE_SIZE, --image_size IMAGE_SIZE IMAGE_SIZE
                        Data image dimensions. Must match model input image dimension (default: 180 320)
  --model_name MODEL_NAME
                        Name of json file where the model is stored (default: model.json)
```

### [`classifier_comparison.py`](code/classifier_comparison.py)
```
usage: classifier_comparison.py data_dir [-h]

positional arguments:
  data_dir              Data Directory
```

## Citation
If you use our code for your paper or work, please cite:
> Puchalla, J., Serianni, A. & Deng, B. Zebrafish identification with deep CNN and ViT architectures using a rolling training window. Sci Rep 15, 8580 (2025). https://doi.org/10.1038/s41598-025-86351-x
```bibtex
@article{puchalla_zebrafish_2025,
	title = {Zebrafish identification with deep {CNN} and {ViT} architectures using a rolling training window},
	volume = {15},
	issn = {2045-2322},
	url = {https://doi.org/10.1038/s41598-025-86351-x},
	doi = {10.1038/s41598-025-86351-x},
	number = {1},
	journal = {Scientific Reports},
	author = {Puchalla, Jason and Serianni, Aaron and Deng, Bo},
	month = mar,
	year = {2025},
	pages = {8580},
}
```

## Acknowledgements
This publication was supported by the Princeton University Library Open Access Fund. The experiments presented in this work were performed on computational resources managed and supported by Princeton Research Computing, a consortium of groups including the Princeton Institute for Computational Science and Engineering PICSciE and Research Computing at Princeton University.
