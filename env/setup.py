# setup.py
import torch
import os

os.chdir('/content')

# Change Pytorch Version
os.system('pip install -q torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113')

# Install mmcv
os.system('pip install -qq mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html')

# Install mmdet
os.system('git clone https://github.com/open-mmlab/mmdetection')
os.chdir('mmdetection')
os.system('pip install -e .') 

# Install Wandb
os.system('pip install -q --upgrade wandb')
import wandb
wandb.login()

# Setting System Path
sys.path.append('/content/drive/MyDrive/Air_PY')

import mmdet
print(f"✨ Pytorch Version : {torch.__version__} \n")
print(f"✨ MMDet Version : {mmdet.__version__} \n")
print(f"✨ Wandb Version : {wandb.__version__} \n") 