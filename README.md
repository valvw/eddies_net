## Software Requirements

- Python 3.9–3.11
- pip version 19.0 or higher for Windows
- Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017, and 2019 
### NVIDIA software for GPU support
- NVIDIA GPU drivers version 450.80.02 or higher
- [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- [cuDNN SDK 8.1.1](https://developer.nvidia.com/rdp/cudnn-archive)

## Installation Steps:
1. Install Miniconda 
Download and install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html).
2. Create a Conda Environment: 
Run Miniconda CLI
```
conda create --name tf python=3.9
conda activate tf
```
3. GPU setup 
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```
4. Install TensorFlow 
```
pip install "tensorflow<2.11" 
```
5. Verify the installation 
```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
If a list of GPU devices is returned, you've installed TensorFlow successfully.

## Requirements
Install the packages:
```
pip install -r requirements.txt
```
## Load this repository
Click on the "Code" button located near the top right of the repository page. In the dropdown menu that appears, click on the "Download ZIP" option. 
Unzip  downloaded-archive.zip -d /path/to/user/../  ->  cd extracted-folder-name

## Load and unzip dataset 
```
!pip install --upgrade --no-cache-dir gdown
gdown --fuzzy 'https://drive.google.com/file/d/1CUMTMqwxq3bfabRgerlY3bufDcomVmYq/view?usp=sharing'
!unzip patches_i.zip
```
```
gdown --fuzzy 'https://drive.google.com/file/d/1X861Y1okQ944zqnRTtq8NxPhUmnb4uMs/view?usp=sharing'
!unzip patches_m3.zip
```
## Project Structure
Check!!
```
eddies_net/
│
├── data/
│   ├── patches_i
│   ├── patches_m
│
├── loader.py
│
├── net.py
│
├── requirements.txt
│
├── README.md
```
## Training Specification
batch size : 16 - 32  
number of epochs : 30 - 50 - 100 

## Train
Use python net.py to start the training !(trained model will be saved)
