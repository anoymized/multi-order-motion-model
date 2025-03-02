
# Machine Learning Modeling for Multi-Order Human Visual Motion Processing

This repository contains code and datasets for our research on developing machine learning models that mimic human visual motion perception. While state-of-the-art computer vision (CV) models, such as deep neural networks (DNNs), excel at estimating optical flow in naturalistic images, they often fall short of replicating the biological visual system’s ability to perceive **second-order motion** (i.e., motion of higher-order image features). Our biologically inspired approach bridges this gap by proposing a model architecture aligned with psychophysical and physiological findings.

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Requirements](#system-requirements)
  - [Hardware Requirements](#hardware-requirements)
  - [Software Requirements](#software-requirements)
    - [OS Requirements](#os-requirements)
    - [Python Dependencies](#python-dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [Testing the Model](#testing-the-model)
  - [Training the Model](#training-the-model)
  - [Model Inference and Evaluation](#model-inference-and-evaluation)
- [Human Data and Second-Order Benchmark](#human-data-and-second-order-benchmark)
- [Data Render and Motion Data Generation](#data-render-and-motion-data-generation)
- [License](#license)

---

## Overview

Human visual systems exhibit remarkable robustness when interpreting complex motion cues, including those involving non-Lambertian materials (e.g., glossy, specular, or translucent surfaces). Inspired by the human V1–MT motion processing pathway, our model integrates:

1. **Trainable Motion Energy Sensor Bank**  
   - Mimics first-order (luminance-based) motion extraction.
2. **Recurrent Graph Network**  
   - Captures dynamic dependencies for robust motion representation.
3. **Second-Order Motion Pathway**  
   - Utilizes nonlinear preprocessing through a naive 3D CNN block.

This hybrid approach, which combines biologically plausible mechanisms with modern deep learning, achieves robust object motion estimation and generalizes well to both first- and second-order motion phenomena.

---

## Key Features

- **First- and Second-Order Motion Processing**  
  Handles both luminance-based and nonlinear feature-based motion cues.
- **Biological Alignment**  
  Inspired by the cortical V1–MT motion pathway, aligning with psychophysical and physiological data.
- **Robust Object Motion Estimation**  
  Trained on datasets featuring non-Lambertian materials, ensuring robustness to optical artifacts.
- **Novel Datasets**  
  Includes custom-designed motion datasets with varied material properties for moving objects.

---

## System Requirements

### Hardware Requirements
- A CUDA and cuDNN-supported GPU is strongly recommended due to the group-wise convolution used in the motion energy module.
- Running on CPU alone may be extremely time-consuming.

### Software Requirements

#### OS Requirements
- **Windows:** 11  
- **Linux:** Ubuntu 20.04

#### Python Dependencies
This implementation primarily depends on the Python scientific stack:

```
torch                         2.0.0
torchvision                   0.15.0
numpy                         1.23.5
opencv-python                 4.2.0.34
scipy                         1.10.1
matplotlib                    3.7.5
easydict                      1.10
processbar                    1.0.8
imageio                       2.26.0
imageio-ffmpeg                0.4.8
path                          16.2.0
pydensecrf                    1.0
Pillow                        9.4.0
pingouin                      0.5.3
scikit-image                  0.20.0
scikit-learn                  1.2.2
seaborn                       0.13.2
pandas                        2.0.0
fast-slic                     0.4.0
```

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/anoymized/multi-order-motion-model.git
   cd multi-order-motion-model
   ```
2. **Set Up a Conda Environment (Recommended)**:
   ```bash
   conda create -n motion_env python=3.9
   conda activate motion_env
   ```
3. **Install Dependencies**:
   We recommend installing each dependency one by one to avoid version conflicts:
   ```bash
   pip install torch==2.0.0 torchvision==0.15.0
   pip install numpy==1.23.5 opencv-python==4.2.0.34 scipy==1.10.1 ...
   ```
   *Continue installing until all required packages are successfully installed.*

Typical install time: Within 30 minutes.

---

## Usage

### Testing the Model

1. **Download Pretrained Model**  
   Pretrained model checkpoint can be found here:  
   [Hugging Face Space: final_sintel_kitti.pth](https://huggingface.co/datasets/sunana/mateiral-controlled-motion-dataset/blob/main/final_sintel_kitti.pth)

2. **Demo Test Stimuli**  
   A sample test stimulus folder (`demo`) is included. Make sure the model checkpoint is placed (or its path specified) correctly in your environment.

3. **Run Motion Prediction**  
   ```bash
   python infer_motion.py \
       --model path/to/dual_model_final.pth \
       --path path/to/test-stimuli
   ```
   Example defaults in the script are:
   ```python
   parser.add_argument('--model', help="restore checkpoint",
                       default="modelckpt/dual_model_final.pth")
   parser.add_argument('--path', help="dataset for evaluation",
                       default='demo/test-stimuli/segmentation/soapbox')
   ```
   Adjust these arguments (e.g., paths to the model checkpoint and test video folder) based on your local setup.

4. **Run Segmentation**  
   ```bash
   python infer_segmentation.py \
       --model path/to/dual_model_final.pth \
       --path path/to/test-stimuli
   ```
   Similarly, you can edit the `--model` and `--path` parameters for different checkpoints and data paths.

---

### Training the Model

1. **Download the Training Datasets**  
   We provide two mini motion datasets featuring diffuse and non-diffuse objects:  
   [Google Drive: Training Datasets](https://drive.google.com/file/d/1vWx4C_uQI6Dd5Mn9BotCdvOLfn5nj4XN/view?usp=sharing)  
   You can use these to verify the effect of diffuse and non-diffuse data on second-order motion perception.  
   Our full dataset (diffuse data, non-diffuse data, drifting grating, simple non-texture motion) is provided at:  
   [Hugging Face Space](https://huggingface.co/datasets/sunana/mateiral-controlled-motion-dataset)

2. **Update Configuration**  
   - Modify the `configdict.py` file to point to your local dataset paths and adjust hyperparameters as needed.

3. **Run the Training Script**  
   ```bash
   python train_full_model.py
   ```
   The model will automatically save checkpoints and training logs to the specified outputs directory.

---

### Model Inference and Evaluation

After you download our pretrained model, the `evaluate_human` folder contains scripts to generate the model responses. Files such as `infer_kitti2015.py`, `infer_sec_motion.py`, and `infer_sintel_slow.py` are provided for generating model responses on different datasets.

**Note:**
- You must first download the selected KITTI 2015, Sintel Slow, and second-order motion benchmark from Hugging Face:  
  [Hugging Face Space](https://huggingface.co/datasets/sunana/mateiral-controlled-motion-dataset/tree/main)  
  Then deploy the datasets at the correct addresses.
- The selected KITTI dataset is located in `model_response_kitti.zip`.
- MATLAB code for evaluation, data analysis, and human response of Sintel-slow and KITTI2015 is provided in the `evaluate_human/matlab_code` folder.

---

## Human Data and Second-Order Benchmark

- **Data & Model Responses:**  
  Human psychophysical data and the model’s responses for second-order motion are located in the `second-order-exp` folder.
- **Visualization:**  
  Use the provided Jupyter notebooks in the `dataviz` folder to visualize and analyze results.
- **Second-Order Benchmark:**  
  [Hugging Face Space: human_static.zip](https://huggingface.co/datasets/sunana/mateiral-controlled-motion-dataset/tree/main)
- **Psychopy Protocol:**  
  If you wish to run additional psychophysical experiments:
  1. Download the second-order motion benchmark from the link above.
  2. Install `psychopy` (visit [https://www.psychopy.org/download.html](https://www.psychopy.org/download.html)).
  3. Run `TestTrial.py` for a demo or `FormalExp.py` for the full experiment.
  4. Adjust `configure.py` to match your environment settings.

---

## Data Render and Motion Data Generation

The `Data_Generator` folder contains scripts to generate both simple non-texture motion and second-order motion:
- Run `create_nontexture_data.py` to generate simple non-texture motion.
- Run `create_second_order_dataset.py` or `secondorder_human.py` to generate second-order motion.

For rendering material-controlled data, refer to the README and scripts in the `material_data_render` folder.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE). Please see the [LICENSE](LICENSE) file for more details.
