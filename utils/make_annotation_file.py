
# 👉 Annotation 용도로 쓰일 Meta File (.txt) 생성하는 파이썬 파일
# 👉 Train & Val & Test로 분할돼있지 않은 데이터(이미지)의 파일명을 추출해서 파일명을 담은 train & test txt파일 형성

# --> Split dataset into Train & Test   (Split dataset into Train & Val & Test는 ./extract_split_move.py에 위치)
# --> 데이터셋 생성 시 이미지 파일명을 가져와서 이미지 파일에 매칭되는 json 파일로 label 작업


'''
< Structure Example >
├── Root folder (name : airplane_custom)
│   ├── Dataset folder (name : 100_data)
│         ├── images (folder name : 100_images)
│         ├── labels (folder name : 100_labels)
│         ├── train.txt
│         ├── val.txt
│         ├── (text.txt)
'''



###################################### 👉 Import Libraries 👈 ##########################################

# #Install mmcv if not installed
# print(f"✨[Info Msg] MMCV Install Start \n")
# os.system('pip install -qq mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html')
# print(f"✨[Info Msg] MMCV Install Complete 🛠️ \n\n")


import os
import mmcv # mmcv 설치
import argparse
import pandas as pd
from glob import glob
from tqdm import tqdm


###################################### 👉 Hyperparameters 👈 ##########################################
def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract_Split_Move')
    parser.add_argument('--root_dir', help='Path : root_dir/dataset_dir/images or labels.')
    parser.add_argument('--dataset_dir', help='Name of dataset folder.') # 100_data
    parser.add_argument('--test_ratio', type=float, default=0.2, 
                        help='Ratio of test data to generate. Train : 0.8 / Test : 0.2')
    parser.add_argument('--tr_anno_name', help='Name of Train Annotation file.')
    parser.add_argument('--val_anno_name', help='Name of Valid Annotation file.')
    parser.add_argument('--test_anno_name', help='Name of Test Annotation file.')   

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():

    args = parse_args()

    os.chdir(f'{args.root_dir}')
    target_path = f'{args.root_dir}/{args.dataset_dir}/'
    tgt_img_path = os.path.join(target_path, '100_images')
    tgt_label_path = os.path.join(target_path, '100_labels')


    print(f'\n✨[Info msg] Loading total images and labels')
    img_100_list = sorted(tqdm(glob(os.path.join(tgt_img_path,'*'))))
    label_100_list = sorted(tqdm(glob(os.path.join(tgt_label_path,'*'))))
    print(f'total imgs num : {len(img_100_list)}')
    print(f'total labels num : {len(label_100_list)}\n')


    ###################################### 👉 Extract Filename to make Annotation file 👈##########################################
    print(f'\n✨[Info msg] Extract filenames out of imgs & labels list to make Annotation file')
    def name_parsing(file_list):

        name = []

        for i in tqdm(range(len(file_list))):
            file_list[i] = file_list[i].split('/')[-1].split('.')[0]
            name.append(file_list[i])

        return name

    imgs = name_parsing(img_100_list)
    labels = name_parsing(label_100_list)

    print(f'\nlength of imgs: {len(imgs)}')
    print(f'length of labels : {len(labels)}\n')



    ############################################### 👉 Data Split 👈##########################################

    # split the 100_data into two groups
    # trian 0.8, test 0.2

    # sklearn 패키지에서 제공하는 ShuffleSplit과 데이터셋을 분할
    # shufflesplit 함수는 데이터셋 인덱스를 무작위로 사전에 설정한 비율로 분할
    # 즉, 4:1 로 분할하고 싶은 경우에 무작위 인덱스로 4:1 비율로 분할

    print(f'\n✨[Info msg] Split data into Train & Test.. Ratio : {float(1-args.test_ratio)} & {args.test_ratio} ')

    from sklearn.model_selection import ShuffleSplit

    train_idx = []
    test_idx = []

    sss = ShuffleSplit(n_splits=1, test_size=args.test_ratio, random_state=100)
    indices = range(len(imgs)) # 이미지의 총 개수를 인덱스로 변환

    # 각 이미지들의 파일명이 리스트의 인덱스로 접근할 수 있으므로 인덱스를 활용해서 train & test 분할
    for train_index, test_index in sss.split(indices):  
        train_idx.append(train_index)
        test_idx.append(test_index)
    
    '''
    12000
    ----------
    3000 

    train_idx
    -->
    [array([ 6859, 10995,  4306, ..., 14147,  6936,  5640])]
    '''

    ################### 👉 Images Split #######################
    ## images

    # ShuffleSplit은 array로 반환되므로 list로 변환
    tr_idx = train_idx[0].tolist()
    test_idx = test_idx[0].tolist()

    print(f'length of tr_idx :  {len(tr_idx)}')
    print(f'length of test_idx :  {len(test_idx)}\n')


    tr_imgs_list = []
    test_imgs_list = []


    for i in tr_idx :
        tr_img = imgs[i]
        tr_imgs_list.append(tr_img)


    for i in test_idx:
        test_img = imgs[i]
        test_imgs_list.append(test_img)
        
    print(f'Quantity of Train data :  {len(tr_imgs_list)}')
    print(f'Quantity of Test data :  {len(test_imgs_list)}\n')



    ################### 👉 Labels Split #######################
    ## labels

    tr_labels_list = []
    test_labels_list = []

    for i in tr_idx :
        tr_label = labels[i]
        tr_labels_list.append(tr_label)


    for i in test_idx :
        test_label = labels[i]
        test_labels_list.append(test_label)


    print(f'Quantity of Train label :  {len(tr_labels_list)}')
    print(f'Quantity of Test label :  {len(test_labels_list)}\n')


    print('\n✨[Info msg] Check whether they are matched')
    print(f'train : {tr_imgs_list == tr_labels_list}')
    print(f'test : {test_imgs_list == test_labels_list}\n')

    ############################################### 👉 ann_file (meta_file) generation 👈##########################################
    ## Meta File : txt 파일 생성 (Annotation에 쓸 것)

    print('\n✨[Info msg] Start generating annotation file (Meta File) for Middle Format \n')
    # list에 담겨 있는 filename을 활용하여 pd.Dataframe 생성
    tr_df = pd.DataFrame({'tr_filename' : tr_imgs_list})
    test_df = pd.DataFrame({'test_filename': test_imgs_list, 'test_labels_name' : test_labels_list})


    # # 기존 인덱스(annotation 파일) 삭제 
    # # Cross Validation 역할
    # os.system('rm ./100_data/train.txt')
    # os.system('rm ./100_data/val.txt')


    # Annotation 파일용 txt 파일 생성
    # args.root_dir / args.dataset_dir / args.tr or val or test
    tr_df['tr_filename'].to_csv(f'{args.root_dir}/{args.dataset_dir}/{args.tr_anno_name}', index=False, header=False)
    test_df['test_filename'].to_csv(f'{args.root_dir}/{args.dataset_dir}/{args.test_anno_name}', index=False, header=False)
    print('✨[Info msg] Generating annotation file (Meta File) Complete')
    print(f'✨[Info msg] Saving ... --> {args.root_dir}/{args.dataset_dir}/{args.tr_anno_name}')
    print(f'✨[Info msg] Saving ... --> {args.root_dir}/{args.dataset_dir}/{args.test_anno_name} \n')


    # txt 내에 있는 인자들을 리스트 형태로 불러옴
    print('✨[Info msg] Check filenames in the annotation file\n')
    image_tlist = mmcv.list_from_file(f'{args.root_dir}/{args.dataset_dir}/{args.tr_anno_name}')
    image_test_list = mmcv.list_from_file(f'{args.root_dir}/{args.dataset_dir}/{args.test_anno_name}')


    print(f'Length of total Train_imgs names : {len(image_tlist[:])}')
    print(f'Train 5 imgs names : \n {image_tlist[:5]}\n')
    print(f'Length of total Test_imgs names : {len(image_test_list[:])}')
    print(f'Test 5 imgs names : \n {image_test_list[:5]}') 


if __name__ =='__main__':
    main()
