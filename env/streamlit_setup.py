import os

# Install Pytorch+cuda
os.system('pip install -q torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113')
import torch
print(torch.__version__)

# Install MMDetection (MMCV automatically install)
os.system('pip install openmim')
os.system('mim install mmdet==2.25.1')
os.system('git clone https://github.com/open-mmlab/mmdetection')

# install streamlit
os.system('pip install -q streamlit')

# install pyngrok
os.system('pip install -q pyngrok')

# install imageio
os.system('pip install -q imageio==2.4.1')
