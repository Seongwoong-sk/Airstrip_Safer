
# 👉 Ablation Study, Optimal Hyperparameter Seacrching 등을 할 때 시간을 절약하기 위해서 
# 👉 Original Dataset에서 클래스 별로 균등하게 추출하여 임시로 만든 Mini Dataset을 만드는 파이썬 파일
# 👉 하나의 클래스에 기능 별로 함수를 만들어서 모듈화
️
# 1️⃣ DataFrame 내 CLASSES 개수 / 이름 변환
# --> 데이터셋의 정보가 들어있는 Dataframe을 활용하여 기존 데이터셋의 클래스 개수를 줄이고(22 -> 19) 

# 2️⃣ 클래스 별로 100개의 이미지씩 모아서 인덱스를 활용하여 데이터셋을 생성

# 3️⃣ Json (label 파일) 안에 있는 class id 수정

# 4️⃣ 100_data/100_image (Mini-dataset 경로) 에서 각 클래스별 인덱스를 추출해서 그것을 이용해서 데이터 분할

# 5️⃣ Train, Valid용 인덱스 활용해서 Annotation 용 Meta file (txt파일) 생성

'''
< Structure Example >
├── Root folder (name : airplane_custom)
│   ├── Dataset folder (name : 100_data)
│         ├── images (folder name : 100_images)
│         ├── labels (folder name : 100_labels)
│         ├── train.txt
│         ├── val.txt
│         ├── (text.txt)
│   ├── explode_df.csv

'''

###################################### 👉 Libraries 👈 ################################################################

# #Install mmcv if not installed

# print(f"✨[Info Msg] MMCV Install Start \n")
# os.system('pip install -qq mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html')
# print(f"✨[Info Msg] MMCV Install Complete 🛠️ \n\n")

import os
import json
import mmcv
import random
import argparse
import itertools
import pandas as pd
from tqdm import tqdm
from glob import glob
from shutil import copyfile

def parse_args():
    parser = argparse.ArgumentParser(description='Mini-Dataset Generation')
    parser.add_argument('--df', default='/content/drive/MyDrive/airplane_custom/explode_df.csv',
                        help='Path of Dataframe to work on')
    parser.add_argument('--mini_root_dir', default='/content/drive/MyDrive/airplane_custom/100_data/',
                        help='Root_dir of Mini-Dataset')
    parser.add_argument('--original_root_dir', default='/content/drive/MyDrive/airplane/',
                        help='Root_dir of Original-Dataset')
    parser.add_argument('--class_num',type=int, help='Quantity of classes to work on')
    parser.add_argument('--val_ratio',type=float, help='Ratio of Validation Data when Spliting')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    class MDMaker:

        def __init__(self):

            # DataFrame 내 CLASSES 개수 / 이름 변환할 DataFrame
            self.df = args.df
            
            # Mini-Dataset의 root_dir
            self.mini_root_dir = args.mini_root_dir

            # Original-Dataset의 root_dir
            self.original_root_dir = args.original_root_dir

            # Quantity of classes to work on
            self.class_num = args.class_num # 19

            # ratio of validation to split
            self.val_ratio = args.val_ratio # 0.15


    ###################################### 👉 DataFrame 내 CLASSES 개수 / 이름 변환 👈 ##########################################
        '''
        < 기존 AI HUB 데이터셋의 클래스 개수 22 -> 19 >

        기존 CLASSES = [  22개 ::
            'Civil aircraft', 'Military aircraft', 'Rotorcraft', 'Light aircraft', 'Road surface facility',
            'Road surface disorder', 'Obstacle', 'Bird', 'Mammals', 'Worker', 'Box','Pallet', 'Toyinka',
            'Ramp bus', 'Step car', 'Fire truck', 'Road sweeper', 'Weeding vehicle', 'Special vehicle',
            'Forklift', 'Cargo loader', 'Tug Car' ]

        -----------------------------------------------------------------------------------------------------

        변경 후 CLASSES = [ 19개 :: 
            'Aircraft','Road surface facility','Road surface disorder','Obstacle','Bird','Mammals',
            'Worker', 'Box','Pallet','Toyinka','Ramp bus','Step car','Fire truck','Road sweeper',
            'Weeding vehicle', 'Special vehicle','Forklift','Cargo loader','Tug Car']

        --> Civil aircraft + Military aircraft + Rotorcraft + Light aircraft ---->> " Aircraft "
        '''

        def class_name_change(self):

            ### Take a look at classes ###
            print(f'✨[Info msg] Check large_class, middle_class, class in Dataframe')
            df = pd.read_csv(self.df)
            print(f"large class : {sorted(df['large_class'].unique().tolist())}") # 지정 / 이상
            print(f"middle class : { sorted(df['middle_class'].unique().tolist())}") # 항공기, 활주로 / 동물, 사람, 이상 물체, 조업 차량
            print(f"class : {sorted(df['class'].unique().tolist())} \n")

            ### Dataframe 내 Class ID Number 변환 ###
            print(f'\n✨[Info msg] Check large_class, middle_class, class in Dataframe')

            # class 1, 2, 3, 4 --> 1
            change_value_dict = {1 : 1, 2: 1, 3:1 , 4:1}
            print(f'✨[Info msg] Apply change class dict : {change_value_dict}')
            df = df.replace({'class' : change_value_dict})
            print(f"✨[Info msg] Check whether classes that should remove exist or not : {df[df['class'].isin([2,3,4])]}")

            # class 5 ~ 23 per -3  --> 2 ~ 19
            print(f"\n✨[Info msg] Generate dict containing from class 5 to 23 per '-3' ")
            change_rest_dict = {i:i-3 for i in range(5,23,1)}
            print(f'✨[Info msg] Apply change class dict : {change_rest_dict}')
            df = df.replace({'class' : change_rest_dict})
            print(f"✨[Info msg] Check whether classes that should remove exist or not : {df[df['class'].isin([20,21,22])]} \n")


            # 전체 확인
            print(f"✨[Info msg] Check entire classes after processing : {sorted(df['class'].unique())}")
            print(df['class'].value_counts(), '\n')


            ### Dataframe 내 Class Name Replacement matched by Class ID ###
            # class 2,3,4 --> class 1
            print(f"✨[Info msg] Dataframe 내 Class Name Replacement matched by Class ID \n")

            CLASSES = ['Aircraft','Road surface facility','Road surface disorder','Obstacle','Bird','Mammals','Worker',
                        'Box','Pallet','Toyinka','Ramp bus','Step car','Fire truck','Road sweeper','Weeding vehicle',
                        'Special vehicle','Forklift','Cargo loader','Tug Car']

            # class id : class name Dictionary 생성
            cat2label = {i+1:k for i,k in enumerate(CLASSES)}

            # class 1인 class name을 Aircraft로 변경
            # 앞에 class 2,3,4를 다 1로 바꿨기 때문에 class id 1인것만 바꾸면 2,3,4를 다 바꾼 것
            df.loc[df['class']== 1, 'class_name'] = 'Aircraft'
            class_list = df['class_name'].unique().tolist()
            print(f"✨[Info msg] Check class names {class_list}\n")

            

    ###################################### 👉 전체 이미지 중에서 Dataframe 활용하여 class 19개 각각 100개씩 모으기 👈 ##########################################
            
            ### 각 클래스가 100개씩 들어있는 이미지 경로가 담긴 리스트 생성 ###
            print(f'✨[Info msg] Start Collecting 100 images per class evenly \n')

            # 무작위 추출로 클래스 19개 100개씩 추출
            def make_cls_list(df,class_num):
                cls_list = random.sample(df.loc[df['class']==class_num, 'image'].tolist(), k=100)
                return cls_list

            # cls_list_1, 2, 3, 4, 5, 6 ,,,,,, 19
            print(f'✨[Info msg] Generating each list consisting of 100 imgs per each class \n')
            for i in range(1,20): 
                globals()['cls_list_'+str(i)] = make_cls_list(df,i) # cls_list_1,2,3,...19 
            # len(cls_list_15) # 100


            ### cls_list_{}에 있는 이미지 경로를 100_data/100_images/ (Mini_Dataset 경로)에 저장 ###
            target_path = self.mini_root_dir
            tgt_img_path = os.path.join(target_path, '100_images')
            tgt_label_path = os.path.join(target_path, '100_labels')
            DATA_PATH = self.original_root_dir

            def movefile(filelist, movelist):
                for idx, filepath in enumerate(tqdm(filelist)):
                    copyfile(filepath, movelist[idx])
            
            # copyfile
            print(f'✨[Info msg] Move images per class to Mini-Dataset Path \n')
            for i in range(1,20): # class 1 ~ 19

                # class별 list를 하나씩 인자로 입력해서 리스트 내 요소들이 하나씩 이동함
                # Image data Movement
                print("[Image] Train class id : {}".format(i))
                movefile([os.path.join(filename) for filename in globals()['cls_list_'+str(i)]], [os.path.join(tgt_img_path,'Cls_'+ str(i) +'_'+ filename.split('/')[-1]) for filename in globals()['cls_list_'+str(i)]])

                # Label Movement
                print("[Label] Train class id : {}".format(i))
                movefile([os.path.join(DATA_PATH, 'label', filename.split('/')[-1].split('.')[0]+'.json') for filename in globals()['cls_list_'+str(i)]],\
                        [os.path.join(tgt_label_path, 'Cls_'+ str(i) +'_'+ filename.split('/')[-1].split('.')[0]+'.json') for filename in globals()['cls_list_'+str(i)]])


            ### 클래스가 중복된 이미지가 있어서 개수 감소가 일어남. --> 총 1,889개 ###
            img_100_list = sorted(tqdm(glob(os.path.join(tgt_img_path,'*'))))
            label_100_list = sorted(tqdm(glob(os.path.join(tgt_label_path,'*'))))
            print(f'✨[Info msg] Total Quantity of Mini-Dataset Data / Label --> {len(img_100_list)}/{len(label_100_list)} \n')
    

        
    ###################################### 👉 Json (label 파일) 안에 있는 class id 수정 👈 ##########################################
        '''
        < 원본 json 파일의 annotation 형식 >
        ----->  {'data ID': 'S1', 'middle classification': '06', 'flags': {}, 'box': [1455, 776, 1772, 871], 'class': '13', 'large classification': '02'}
                {'data ID': 'S1', 'middle classification': '04', 'flags': {}, 'box': [1678, 728, 1740, 858], 'class': '10', 'large classification': '02'}

        < Class id 수정 후 >
        ---->   {'data ID': 'S1', 'middle classification': '06', 'flags': {}, 'box': [1455, 776, 1772, 871], 'class': 10, 'large classification': '02'}
                {'data ID': 'S1', 'middle classification': '04', 'flags': {}, 'box': [1678, 728, 1740, 858], 'class': 7, 'large classification': '02'}
  
        '''
        
        def class_id_change(self):

            target_path = self.mini_root_dir
            tgt_label_path = os.path.join(target_path, '100_labels')
            label_100_list = sorted(tqdm(glob(os.path.join(tgt_label_path,'*'))))
            print(f'✨[Info msg] Change class ids in json file \n --> its location : {tgt_label_path} \n')

            # Class id 2,3,4 --> 1  //  Class id 5 ~ 23 --> 각각 -3 
            def class_change_json(file_path):

                for i in tqdm(range(len(file_path)), desc= 'class id chaning per json file'): 

                    with open(file_path[i],'r') as f:
                        jsonFile = json.load(f)

                        for ann_data in jsonFile['annotations']:
                            ann_data['class'] = int(ann_data['class'])

                            if ann_data['class'] in [2,3,4] :
                                ann_data['class'] = 1

                            elif ann_data['class'] == 1:
                                ann_data['class'] = 1

                            else : ann_data['class'] -= 3

                    # 기존 json 파일 덮어쓰기
                    with open(file_path[i], 'w', encoding='utf-8') as file:
                        json.dump(jsonFile,file, indent='\t')

            class_change_json(label_100_list)
            print(f'✨[Info msg] Change class ids in json file Complete \n')



    ###################################### 👉 100_data/100_image (Mini-dataset 경로) 에서 각 클래스별 인덱스를 추출해서 그것을 이용해서 데이터 분할 👈 ##########################################
        def extract_indexes_per_class(self):


            # 각 클래스별 0.15 ratio로 Train / Val Split
            print(f'✨[Info msg] Split data into Train and Valid')

            ### 파일명 추출 ###
            print(f'✨[Info msg] Extract file_names \n')
            target_path = self.mini_root_dir
            tgt_img_path = os.path.join(target_path, '100_images')
            tgt_label_path = os.path.join(target_path, '100_labels')


            img_100_list = sorted(tqdm(glob(os.path.join(tgt_img_path,'*'))))
            label_100_list = sorted(tqdm(glob(os.path.join(tgt_label_path,'*'))))
            print(f'Total imgs num : {len(img_100_list)}')
            print(f'Total labels num : {len(label_100_list)}\n')


            def name_parsing(file_list):

                list_ = []

                for i in tqdm(range(len(file_list))):
                    file_list[i] = file_list[i].split('/')[-1].split('.')[0]
                    list_.append(file_list[i])

                return list_

            imgs = sorted(name_parsing(img_100_list))
            labels = sorted(name_parsing(label_100_list))

            print(f'\nlength of img file_names: {len(imgs)}')
            print(f'length of label file_names : {len(labels)}\n')


            ### class 별 이미지 개수 확인
            print(f'✨[Info msg] Check Quantity of Images per Class \n')


            quantity = []

            for i in range(1,(self.num_class+1),1):

                globals()['img_cl_'+str(i)] = []

                # img_cl_1 , 2, 3, 4, ,,,,,19
                for name in imgs:
                    if name.startswith('Cls_{}_'.format(str(i))):
                        
                        globals()['img_cl_'+str(i)].append(name)
                        num = len(globals()['img_cl_'+str(i)])
                
                quantity.append(num)
                    
            numofCat = {i+1:k for i,k in enumerate(quantity)}
            print(f'✨[Info msg] Check Data Quanitty per class \n {numofCat}')
            '''
            { 1: 100, 
            2: 100, 
            3: 97, 
            4: 100, 
            5: 98, 
            6: 100,
            7: 100, 
            8: 100, 
            9: 91, 
            10: 100, 
            11: 97, 
            12: 100,
            13: 100,
            14: 107, 
            15: 100, 
            16: 100, 
            17: 100, 
            18: 100, 
            19: 99 }
            '''

            ### Validation Ratio 설정과 그에 따른 validation data 수량 추출 ###
            val_ratio = [int(numofCat.get(i)* self.val_ratio) for i in range(1,(self.class_num+1))]
            print(f'✨[Info msg] Validation ratio : {val_ratio} and Validation Data Quantity along this ratio ---> \n {val_ratio} \n length : {len(val_ratio)}')


            ### Class 별 리스트 내 인덱스 추출
            # --> imgs 기준 각 클래스별 인덱스 시작 위치
            indexes = {1 : 1003,
                    2: 1103,
                    3 : 1203,
                    4 : 1300,
                    5 : 1400,
                    6 : 1498,
                    7 : 1598,
                    8 : 1698,
                    9 : 1798,
                    10 : 0,
                    11 : 100,
                    12 : 197,
                    13 : 297,
                    14 : 397,
                    15 : 504,
                    16 : 604,
                    17 : 704,
                    18 : 804,
                    19 : 904}


            ### Train, Valid용 인덱스 생성해서 class 별 균등한 비율로 Data Split 수행 ###

            # split the 100_data into two groups
            # trian 0.85, val 0.15
            # imgs, labels


            # val_ratio : 0.15
            def making_each_class_idx(total_list, class_num, indexes_dict, val_ratio_list,img_quantity_dict,mode):
                '''
                < args >
                total_list : class 별로 데이터를 모아놓은 데이터셋을 담은 list
                class_num : 인덱스를 생성할 클래스 숫자 (int)
                indexes_dict : imgs 기준 각 클래스별 인덱스 시작 위치가 담겨 있는 dict
                val_ratio_list : 클래스별 데이터에 val_ratio를 곱해서 나온 수량이 담겨 있는 list
                img_quantity_dict : 클래스별 데이터 수량이 담겨 있는 dict
                mode : train or valid 
                '''

                train_idx = []
                valid_idx = []
                print(f'Total Index length :: {len(total_list)}')
                print(f'Total number of images of class <{class_num}> is {img_quantity_dict.get(class_num)}')
                key_dict = {x:x for x in range(1,20)}


                if mode == 'valid':
                    
                    # class가 일치할 때

                    if int(class_num) == int(key_dict.get(class_num)):
                        print(f'class <<{class_num}>> appending into VALID_list... ')
                        
                        # class start_index : class_start_index + val_ratio
                        for i in tqdm(range(indexes_dict.get(class_num), indexes_dict.get(class_num)+val_ratio_list[class_num-1],1), desc='Making ★ Valid ★ Indexes....', leave=True): 
                            valid_idx.append(i)
                                
                        print('\n Valid Index')
                        print(f'Idx :: Start_index <{indexes_dict.get(class_num)}> --> End_index <{(indexes_dict.get(class_num)+val_ratio_list[class_num-1]-1)}>')
                        print(f'Class <{class_num}> Train Start Index : <{indexes_dict.get(class_num) + val_ratio_list[class_num-1]}>')
                        print(f'length of valid list : {len(valid_idx)}')

                        return valid_idx


                    # 클래스 19의 다음 클래스는 없으므로 별도 처리
                    elif int(class_num) == int(key_dict.get(19)):

                        #  898  ~ 898 * 0,15 까지
                        for i in tqdm(range(indexes_dict.get(class_num) ,indexes_dict.get(class_num) + val_ratio_list[class_num-1]),indexes_dict.get(1), leave=True):

                            valid_idx.append(i)

                        print('Valid Index')
                        print(f'Idx :: Start_index <{indexes_dict.get(class_num)}> --> End_index <{indexes_dict.get(class_num)+val_ratio_list[class_num-1]-1}>')
                        print('Next Class Index :  NONE')
                        print(f'length of valid list : {len(valid_idx)}')
                        print('DOOONNNNNEEEEEEE!!!!!', '\n')
                                
                        return valid_idx


                if mode == 'train' : 

                    # TypeError: 'int' object is not subscriptable는 인덱스를 갖지않는 값에 인덱스를 가지게 코드를 짤 경우 발생하는 오류이다.
                    # class가 일치할 때
                    if int(class_num) == int(key_dict.get(class_num)):
                        print(f'class <<{class_num}>> appending into TRAIN_list... ')

                        # class = 19일 때는
                        if int(class_num) == int(key_dict.get(19)):

                            #  (valid_idx+1) ~ 996 (class 1 idx-1)까지
                            for i in tqdm(range(indexes_dict.get(class_num) + val_ratio_list[(class_num-1)], indexes_dict.get(1)), leave=True):
                                train_idx.append(i)

                            print('Train Index')
                            print(f'Idx :: Start_Index <{indexes_dict.get(class_num)+val_ratio_list[class_num-1]}> --> End_Index <{(indexes_dict.get(1)-1)}>')
                            print('Next Class Index :  NONE')
                            print(f'length of train list : {len(train_idx)}')
                            print('DOOONNNNNEEEEEEE!!!!!', '\n')

                            return train_idx


                        # class 10의 시작 인덱스가 0이므로 class = 9일 때는 별도 처리       
                        if int(class_num) == int(key_dict.get(9)):
                            gap = img_quantity_dict.get(class_num) - val_ratio_list[class_num-1] # 78
                        
                            for i in tqdm(range(indexes_dict.get(class_num) + val_ratio_list[(class_num-1)], (indexes_dict.get(class_num) + val_ratio_list[(class_num-1)]+gap)), leave=True):
                                train_idx.append(i)

                            print('Train Index')
                            print(f'Idx :: Start_Index <{indexes_dict.get(class_num)+val_ratio_list[class_num-1]}> --> End_Index <{(indexes_dict.get(class_num) + val_ratio_list[(class_num-1)]+gap-1)}>')
                            print(f'Next Class Index :  <{indexes_dict[class_num+1]}>')
                            print(f'length of train list : {len(train_idx)}')
                            print('DOOONNNNNEEEEEEE!!!!!', '\n')

                            return train_idx

                        # 정상일 때는    
                        # class_start_index + val_ratio : next_class_index-1
                        for i in tqdm(range(indexes_dict.get(class_num) + val_ratio_list[class_num-1] , indexes_dict.get(class_num+1)), desc = 'Making ★ Train ★ Indexes.....', leave=True):
                            train_idx.append(i)
                            
                        print('\n Train Index')
                        print(f'Idx :: Start_Index <{indexes_dict.get(class_num)+val_ratio_list[class_num-1]}> --> End_Index <{(indexes_dict.get(class_num+1)-1)}>')
                        print(f"Next Class Index : <{indexes_dict[class_num+1]}>")
                        print(f'length of train list : {len(train_idx)}')
                            
                        return train_idx

                

            # 클래스 별 인덱스를 만들어서 하나의 인덱스에 최종적으로 생성
            def Total_Index_Making(quantity):
                '''
                < args > 
                quantity : index를 만들 클래스의 (끝 숫자 + 1)  ex : 20 --> 1 ~ 19 
                '''

                print(f'Making TOTAL INDEX LIST OF NUMBER OF <{(quantity-1)}> CLASSES \n')
                total_tr_img_idxes = []
                total_tr_label_idxes = []
                total_val_img_idxes = []
                total_val_label_idxes = []

                # class num을 받아서 class 별 인덱스 만드는 함수의 인자로 입력
                for class_num in range(1,quantity,1):

                    val_img_idx = making_each_class_idx(imgs,class_num,indexes, val_ratio,numofCat,mode='valid')
                    val_label_idx = making_each_class_idx(labels,class_num,indexes, val_ratio,numofCat,mode='valid')
                    tr_img_idx = making_each_class_idx(imgs,class_num,indexes, val_ratio,numofCat, mode='train')
                    tr_label_idx = making_each_class_idx(labels,class_num,indexes, val_ratio,numofCat, mode='train')

                    total_tr_img_idxes.append(tr_img_idx)
                    total_tr_label_idxes.append(tr_label_idx)
                    total_val_img_idxes.append(val_img_idx)
                    total_val_label_idxes.append(val_label_idx)

                    print('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★')
                    print(f'CHECK WHETHER IMG LIST AND LABEL LIST OF CLASS <<{class_num}>> ARE MATCHED')
                    print('Train INDEXES of Imgs and Labels are matched ? :: <<{}>> '.format(sorted(tr_img_idx)==sorted(tr_label_idx)))
                    print('Valid INDEXES of Imgs and Labels are matched ? :: <<{}>> '.format(sorted(val_img_idx)==sorted(val_label_idx)))
                    print('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ \n\n\n')

                return total_tr_img_idxes, total_tr_label_idxes, total_val_img_idxes, total_val_label_idxes

            # class 1 ~ 19 train imgs, labels / valid imgs, labels의 인덱스들이 각각 저장됨.    
            tr_img_idx, tr_label_idx , val_img_idx, val_label_idx = Total_Index_Making((self.num_class+1))

            Images = sorted(name_parsing(img_100_list))
            Labels = sorted(name_parsing(label_100_list))
        

    ###################################### 👉 Train, Valid용 인덱스 활용해서 Annotation 용 Meta file (txt파일) 생성 👈 ##########################################
        def gen_annotation(self):

            print(f'\n ✨[Info msg] Start Generating Meta file (.txt) for Annotation using the indexes \n')

            def flatten_list(target_list):
                target_list = itertools.chain(*target_list)
                target_list_ = list(target_list)

                return target_list_

            # 2차원 list로 되어 있어서 1차원으로 변경
            tr_img_idx = flatten_list(tr_img_idx)
            tr_label_idx = flatten_list(tr_label_idx)
            val_img_idx = flatten_list(val_img_idx)
            val_label_idx = flatten_list(val_label_idx)

            global Images, Labels
            print(f'✨[Info msg] Check Quantity of the Indexes')
            print(f'Image Difference (Should be 0) : {len(Images) - len(tr_img_idx) - len(val_img_idx)}')
            print(f'Label Difference (Should be 0) : {len(Labels) - len(tr_label_idx) - len(val_label_idx)}')

            print(f'Train Data Quantity : {len(tr_img_idx)}')
            print(f'Train Label Quantity : {len(tr_label_idx)}')
            print(f'Valid Data Quantity : {len(val_img_idx)}')
            print(f'Valid Label Quantity : {len(val_label_idx)}')


            # 1차원으로 된 리스트 안의 요소를 추출하여 최종 요소가 담긴 리스트 생성
            def final_idx_list(idx, img_or_label):
                idx_list = []

                for num in idx:
                    element = img_or_label[num]
                    idx_list.append(element)
                return idx_list


            tr_imgs_list = final_idx_list(tr_img_idx,Images) # imgs[0] : Cls_10_S1-N01115M00004
            val_imgs_list = final_idx_list(val_img_idx,Images)
            print(f'tr_imgs_num :  {len(tr_imgs_list)}')
            print(f'val_imgs_num :  {len(val_imgs_list)}\n')

            tr_labels_list = final_idx_list(tr_label_idx, Labels)
            val_labels_list = final_idx_list(val_label_idx, Labels)
            print(f'tr_labels_num :  {len(tr_labels_list)}')
            print(f'val_labels_num :  {len(val_labels_list)}\n')

            print('★★★ Check whether they are matched..... ★★★')
            print(f'train : {tr_imgs_list == tr_labels_list}')
            print(f'valid : {val_imgs_list == val_labels_list}\n\n')


            ### ann_file (meta_file) generation ###
            print(f'✨[Info msg] Annotation file (meta_file) Generation')
            os.system('cd /content/drive/MyDrive/airplane_custom')
            tr_df = pd.DataFrame({'tr_filename' : tr_imgs_list})
            val_df = pd.DataFrame({'val_filename': val_imgs_list, 'val_labels_name' : val_labels_list})

            # # 기존 인덱스 삭제 
            # # Cross Validation 역할
            # os.system('rm ./100_data/train.txt')
            # os.system('rm ./100_data/val.txt')

            tr_df['tr_filename'].to_csv(f'{self.mini_root_dir}/train.txt', index=False, header=False)
            val_df['val_filename'].to_csv(f'{self.mini_root_dir}/val.txt', index=False, header=False)
            print(f"✨[Info msg] Saving files....to {f'{self.mini_root_dir}/'}")

            image_tlist = mmcv.list_from_file(f'{self.mini_root_dir}/train.txt')
            image_vlist = mmcv.list_from_file(f'{self.mini_root_dir}/val.txt')

            print(f'Length of total tr_imgs names : {len(image_tlist[:])}')
            print(f'tr_imgs names : \n {image_tlist[:5]}\n')
            print(f'Length of total val_imgs names : {len(image_vlist[:])}')
            print(f'val_imgs names : \n {image_vlist[:5]}') 

            print(f"\n\n ✨[Info msg] Entire Process has done :)")

    # 실행
    mdmaker = MDMaker()
    mdmaker.class_name_change()
    mdmaker.class_id_change()
    mdmaker.extract_indexes_per_class()
    mdmaker.gen_annotation()


if __name__ == '__main__':
    main()
