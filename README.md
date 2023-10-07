## Software Requirements

- Python 3.9–3.11
- pip version 19.0 or higher for Windows
- Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017, and 2019 
### NVIDIA software for GPU support
- NVIDIA GPU drivers version 450.80.02 or higher
- [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- [cuDNN SDK 8.6.0](https://developer.nvidia.com/rdp/cudnn-archive)

## Installation Steps:
1. Install Miniconda 
Download and install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html).
2. Create a Conda Environment: 
Run Miniconda
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
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
If a list of GPU devices is returned, you've installed TensorFlow successfully.