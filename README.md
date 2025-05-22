# SiD Video Distilation
Implementation of Score Identity Distilation [paper](https://arxiv.org/pdf/2404.04057) for CogVideoX and Wan family of video diffusion models.

# Set up Instructions
### Create new environment (change to your package manager as needed)
conda create -n sid_vid python=3.12
conda activate sid_vid

### Install CUDA toolkit first (change to your version as needed)
conda install -c nvidia cuda=12.4.0
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia


### Rest of dependencies
pip install transformers
pip install lightning
pip install peft
pip install wandb
pip install opencv-python

### Since support for Wan is relatively new
pip install git+https://github.com/huggingface/diffusers
pip install git+https://github.com/modelscope/DiffSynth-Studio
