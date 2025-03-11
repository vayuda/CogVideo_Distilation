# Create new environment (change to your package manager as needed)
conda create -n cogvideo python=3.12
conda activate cogvideo

# Install CUDA toolkit first (change to your version as needed)
conda install -c conda-forge cudatoolkit=12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia


# Rest of dependencies
conda install transformers
conda install diffusers
conda install lightning
conda install peft
conda install wandb
conda install PyYAML
conda install numpy
conda install opencv-python
