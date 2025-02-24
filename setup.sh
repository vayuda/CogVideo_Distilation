# Create new environment (change to your package manager as needed)
conda create -n cogvideo python=3.12
conda activate cogvideo

# Install CUDA toolkit first (change to your version as needed)
conda install -c conda-forge cudatoolkit=12.4

# Install PyTorch ecosystem via pip (for 2.6+)
pip install torch torchvision torchaudio

# Rest of dependencies
pip install deepspeed
pip install transformers
pip install diffusers
pip install lightning
pip install peft
pip install wandb
pip install PyYAML
pip install numpy
pip install Pillow
