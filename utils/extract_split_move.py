
# 👉 AI Hub에서 다운로드 받아서 전처리된 데이터와 라벨 데이터들의 수량을 다듬고 
# 👉 " train & val & test "로 분리해서 Original 데이터셋에 옮기는 코드
'''
< Structure Example >
├── Original 데이터셋
│   ├── train
│         ├── train_data
│         ├── train_label
│   ├── valid
│         ├── valid_data
│         ├── valid_label
│   ├── test
│         ├── test_data
│         ├── test_label

├── 작업한 데이터셋
│   ├── images
│   ├── labels
'''

import os
import random
import shutil
import argparse
from tqdm import tqdm
from glob import glob
from shutil import copyfile

###################################### 👉Hyperparameters👈 ##########################################
def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract_Split_Move')
    parser.add_argument('--work_dir', help='Path : work_dir/images or work_dir/labels')
    parser.add_argument('--target_dir', help='Path to original dataset : target_dir/train/train_data')
    parser.add_argument('--quantity', help='Quantity of Data to generate out of Original data')
    parser.add_argument('--tr_ratio', type=float, defaul=0.6, help='Ratio of Train data out of Original data to split')
    parser.add_argument('--split_ratio', type=float, defaul=0.5, help='Ratio to split data out of Rest data :: val & test')
   
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args
    



###################################### 👉 Extract 👈##########################################
def main():

    args = parse_args()
    ## Original 데이터셋에 옮길 작업한 데이터의 수량 확인
    print(f"\n✨[Info msg] Checking Quantity of Working Data")
    imgpaths = sorted(list(glob(f'{args.work_dir}/images/*')))
    labelpaths = sorted(list(glob(f'{args.work_dir}/labels/*')))

    print(f"Quantity of Data : {len(imgpaths)}")
    print(f"Quantity of Label : {len(labelpaths)}")


    ## Data와 대응되지 않는 label 파일이 있으면 label 파일 삭제 -> vice versa도 가능
    print(f"\n✨[Info msg] Extracting and Removing Non-matched Data")
    file_source = f'{args.work_dir}/images/'
    label_source = f'{args.work_dir}/labels/'

    get_files = os.listdir(file_source)
    label_files = os.listdir(label_source)

    for label_filename in tqdm(label_files):

        label_filename = label_filename.split('.')[0]

        if label_filename +'.jpg' not in get_files:
            os.remove(label_source + label_filename+'.json')


    ## Extract 후 수량 확인
    print(f"\n✨[Info msg] Checking Quantity of Working Data After Extracting")
    imgpaths = sorted(list(glob(f'{args.work_dir}/images/*')))
    labelpaths = sorted(list(glob(f'{args.work_dir}/labels/*')))
    print(f"Quantity of Data After Extracting: {len(imgpaths)}")
    print(f"Quantity of Label After Extracting : {len(labelpaths)}")


    ######################################👉 Split 👈##########################################
    ROOT_PATH = args.work_dir
    DATA_PATH = os.path.join(ROOT_PATH, "images")
    LABEL_PATH = os.path.join(ROOT_PATH, "labels")

    TARGET_PATH = args.target_dir

    all_data = os.listdir(os.path.join(DATA_PATH))
    num_extract = args.quantity
    train_num = int(num_extract * args.tr_ratio)
    print(len(all_data), train_num)

    data = random.sample(all_data,k=num_extract)
    train_data = random.sample(data, k= train_num)

    rest = set(data) - set(train_data)

    from sklearn.model_selection import train_test_split

    valid_data, test_data = train_test_split(list(rest), test_size=args.split_ratio, random_state=42)

    print("전체 데이터 갯수 : {} , train 갯수 : {}, val 갯수 : {}, test 갯수 : {}".format(len(data),len(train_data), len(valid_data), len(test_data)))

    # 개수 확인
    if len(data) - len(train_data) - len(valid_data) - len(test_data) != 0:
        exit(0)

    def movefile(filelist, movelist):
        for idx, filepath in enumerate(tqdm(filelist)):
            copyfile(filepath, movelist[idx])

    # copyfile(filepath, movepath)
    # Train 
    print("[Image] Train")
    movefile([os.path.join(DATA_PATH, filename) for filename in train_data], [os.path.join(TARGET_PATH, 'train_data', filename) for filename in train_data])
    print("[Label] Train")
    movefile([os.path.join(LABEL_PATH, filename.split('.')[0]+'.json') for filename in train_data], [os.path.join(TARGET_PATH, 'train_label', filename.split('.')[0]+'.json') for filename in train_data])

    # Val
    print("[Image] Valid")
    movefile([os.path.join(DATA_PATH, filename) for filename in valid_data], [os.path.join(TARGET_PATH, 'valid_data', filename) for filename in valid_data])
    print("[Label] Valid")
    movefile([os.path.join(LABEL_PATH, filename.split('.')[0]+'.json') for filename in valid_data], [os.path.join(TARGET_PATH, 'valid_label', filename.split('.')[0]+'.json') for filename in valid_data])

    # Test
    print("[Image] Test")
    movefile([os.path.join(DATA_PATH, filename) for filename in test_data], [os.path.join(TARGET_PATH, 'test_data', filename) for filename in test_data])
    print("[Label] Test")
    movefile([os.path.join(LABEL_PATH, filename.split('.')[0]+'.json') for filename in test_data], [os.path.join(TARGET_PATH, 'test_label', filename.split('.')[0]+'.json') for filename in test_data])



    ### Original 데이터셋에 옮기고 난 후 수량 확인
    print(f"\n✨[Info msg] Checking Quantity of Original Data")
    tr_img = sorted(list(glob(f'{args.target_dir}/train/train_data/*')))
    print(len(tr_img))
    tr_lab = sorted(list(glob(f'{args.target_dir}/train/train_label/*')))
    print(len(tr_lab))

    val_img = sorted(list(glob(f'{args.target_dir}/valid/valid_data/*')))
    print(len(val_img))
    val_lab = sorted(list(glob(f'{args.target_dir}/valid/valid_label/*')))
    print(len(val_lab))

    test_img = sorted(list(glob(f'{args.target_dir}/test/test_data/*')))
    print(len(test_img))
    test_lab = sorted(list(glob(f'{args.target_dir}/test/test_label/*')))
    print(len(test_lab))

if __name__ == '__main__':
    main()
