# Airstrip Safer

<p align="left">
  <img width="700" src="https://github.com/Seongwoong-sk/Airstrip_Safer/blob/main/github%20main.png" "airstrip">
</p>


One Paragraph of the project description



<br>

## Getting Started


### Prerequisites
1. mmdetection & mmcv 버젼 자동 호환 설치 (Streamlit 사용)
```
pip install openmim
mim install mmdet==2.25.1
```

2. ngrok authentification token 발급 at https://dashboard.ngrok.com/get-started/your-authtoken


### Installation
1. Clone the repo
```
git clone https://github.com/Seongwoong-sk/Airstrip_Safer.git
```
2. Install packages & libraries
```
cd Airstrip_Safer

# local environment
pip install -r requirements.txt
```
or
```
cd Airstrip_Safer

# colab environment
python utils/setup.py
```

<br>

## How to Run
Detail한 arguments들은 각 파일 내에서 확인 가능합니다.

### Train & Validation
```
python tools/train.py \
--config configs/FINAL.py \                        # If initial training, use original_config.py
--work_dir <path to save model weight files> \
--max_epochs <number> \                            # 현재 epoch 기준으로 최대 epoch 설정
--checkpoint <checkpoint_path> \                   # Initial training / model weight path 지정
--resume_from <resume_checkpoint> \                # Resume training / 저장돼있는 model weight path 지정
--seed <number> \
--validate <option> \                              # If not, write False to option blank
--deterministic                                    # Set deterministic options for CUDNN backend 
```


### Test
```
python tools/test.py \
--config configs/FINAL.py \
--checkpoint <path to saved model weight file> \
--work_dir <path to save the file containing evaluation metrics> \
--out <output_filename.pkl> \                   
--eval mAP \
--show_dir <path to show painted images> \
--threshold 0.5
```

### Inference
```
python /content/drive/MyDrive/Air_PY/tools/inference.py \
--config configs/FINAL.py \
--checkpoint <path to saved model weight file> \
--video <path of video to inference> \
--threshold 0.5 \
--out_video <path of result video to save>
```

## Deployment

### 



## Built With

  - [Contributor Covenant](https://www.contributor-covenant.org/) - Used
    for the Code of Conduct
  - [Creative Commons](https://creativecommons.org/) - Used to choose
    the license


