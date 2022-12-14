
# π Ablation Study, Optimal Hyperparameter Seacrching λ±μ ν  λ μκ°μ μ μ½νκΈ° μν΄μ  
# π Original Datasetμμ ν΄λμ€ λ³λ‘ κ· λ±νκ² μΆμΆνμ¬ μμλ‘ λ§λ  Mini Datasetμ λ§λλ νμ΄μ¬ νμΌ
# π νλμ ν΄λμ€μ κΈ°λ₯ λ³λ‘ ν¨μλ₯Ό λ§λ€μ΄μ λͺ¨λν
οΈ
# 1οΈβ£ DataFrame λ΄ CLASSES κ°μ / μ΄λ¦ λ³ν
# --> λ°μ΄ν°μμ μ λ³΄κ° λ€μ΄μλ Dataframeμ νμ©νμ¬ κΈ°μ‘΄ λ°μ΄ν°μμ ν΄λμ€ κ°μλ₯Ό μ€μ΄κ³ (22 -> 19) 

# 2οΈβ£ ν΄λμ€ λ³λ‘ 100κ°μ μ΄λ―Έμ§μ© λͺ¨μμ μΈλ±μ€λ₯Ό νμ©νμ¬ λ°μ΄ν°μμ μμ±

# 3οΈβ£ Json (label νμΌ) μμ μλ class id μμ 

# 4οΈβ£ 100_data/100_image (Mini-dataset κ²½λ‘) μμ κ° ν΄λμ€λ³ μΈλ±μ€λ₯Ό μΆμΆν΄μ κ·Έκ²μ μ΄μ©ν΄μ λ°μ΄ν° λΆν 

# 5οΈβ£ Train, Validμ© μΈλ±μ€ νμ©ν΄μ Annotation μ© Meta file (txtνμΌ) μμ±

'''
< Structure Example >
βββ Root folder (name : airplane_custom)
β   βββ Dataset folder (name : 100_data)
β         βββ images (folder name : 100_images)
β         βββ labels (folder name : 100_labels)
β         βββ train.txt
β         βββ val.txt
β         βββ (text.txt)
β   βββ explode_df.csv

'''

###################################### π Libraries π ################################################################

# #Install mmcv if not installed

# print(f"β¨[Info Msg] MMCV Install Start \n")
# os.system('pip install -qq mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html')
# print(f"β¨[Info Msg] MMCV Install Complete π οΈ \n\n")

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

            # DataFrame λ΄ CLASSES κ°μ / μ΄λ¦ λ³νν  DataFrame
            self.df = args.df
            
            # Mini-Datasetμ root_dir
            self.mini_root_dir = args.mini_root_dir

            # Original-Datasetμ root_dir
            self.original_root_dir = args.original_root_dir

            # Quantity of classes to work on
            self.class_num = args.class_num # 19

            # ratio of validation to split
            self.val_ratio = args.val_ratio # 0.15


    ###################################### π DataFrame λ΄ CLASSES κ°μ / μ΄λ¦ λ³ν π ##########################################
        '''
        < κΈ°μ‘΄ AI HUB λ°μ΄ν°μμ ν΄λμ€ κ°μ 22 -> 19 >

        κΈ°μ‘΄ CLASSES = [  22κ° ::
            'Civil aircraft', 'Military aircraft', 'Rotorcraft', 'Light aircraft', 'Road surface facility',
            'Road surface disorder', 'Obstacle', 'Bird', 'Mammals', 'Worker', 'Box','Pallet', 'Toyinka',
            'Ramp bus', 'Step car', 'Fire truck', 'Road sweeper', 'Weeding vehicle', 'Special vehicle',
            'Forklift', 'Cargo loader', 'Tug Car' ]

        -----------------------------------------------------------------------------------------------------

        λ³κ²½ ν CLASSES = [ 19κ° :: 
            'Aircraft','Road surface facility','Road surface disorder','Obstacle','Bird','Mammals',
            'Worker', 'Box','Pallet','Toyinka','Ramp bus','Step car','Fire truck','Road sweeper',
            'Weeding vehicle', 'Special vehicle','Forklift','Cargo loader','Tug Car']

        --> Civil aircraft + Military aircraft + Rotorcraft + Light aircraft ---->> " Aircraft "
        '''

        def class_name_change(self):

            ### Take a look at classes ###
            print(f'β¨[Info msg] Check large_class, middle_class, class in Dataframe')
            df = pd.read_csv(self.df)
            print(f"large class : {sorted(df['large_class'].unique().tolist())}") # μ§μ  / μ΄μ
            print(f"middle class : { sorted(df['middle_class'].unique().tolist())}") # ν­κ³΅κΈ°, νμ£Όλ‘ / λλ¬Ό, μ¬λ, μ΄μ λ¬Όμ²΄, μ‘°μ μ°¨λ
            print(f"class : {sorted(df['class'].unique().tolist())} \n")

            ### Dataframe λ΄ Class ID Number λ³ν ###
            print(f'\nβ¨[Info msg] Check large_class, middle_class, class in Dataframe')

            # class 1, 2, 3, 4 --> 1
            change_value_dict = {1 : 1, 2: 1, 3:1 , 4:1}
            print(f'β¨[Info msg] Apply change class dict : {change_value_dict}')
            df = df.replace({'class' : change_value_dict})
            print(f"β¨[Info msg] Check whether classes that should remove exist or not : {df[df['class'].isin([2,3,4])]}")

            # class 5 ~ 23 per -3  --> 2 ~ 19
            print(f"\nβ¨[Info msg] Generate dict containing from class 5 to 23 per '-3' ")
            change_rest_dict = {i:i-3 for i in range(5,23,1)}
            print(f'β¨[Info msg] Apply change class dict : {change_rest_dict}')
            df = df.replace({'class' : change_rest_dict})
            print(f"β¨[Info msg] Check whether classes that should remove exist or not : {df[df['class'].isin([20,21,22])]} \n")


            # μ μ²΄ νμΈ
            print(f"β¨[Info msg] Check entire classes after processing : {sorted(df['class'].unique())}")
            print(df['class'].value_counts(), '\n')


            ### Dataframe λ΄ Class Name Replacement matched by Class ID ###
            # class 2,3,4 --> class 1
            print(f"β¨[Info msg] Dataframe λ΄ Class Name Replacement matched by Class ID \n")

            CLASSES = ['Aircraft','Road surface facility','Road surface disorder','Obstacle','Bird','Mammals','Worker',
                        'Box','Pallet','Toyinka','Ramp bus','Step car','Fire truck','Road sweeper','Weeding vehicle',
                        'Special vehicle','Forklift','Cargo loader','Tug Car']

            # class id : class name Dictionary μμ±
            cat2label = {i+1:k for i,k in enumerate(CLASSES)}

            # class 1μΈ class nameμ Aircraftλ‘ λ³κ²½
            # μμ class 2,3,4λ₯Ό λ€ 1λ‘ λ°κΏ¨κΈ° λλ¬Έμ class id 1μΈκ²λ§ λ°κΎΈλ©΄ 2,3,4λ₯Ό λ€ λ°κΎΌ κ²
            df.loc[df['class']== 1, 'class_name'] = 'Aircraft'
            class_list = df['class_name'].unique().tolist()
            print(f"β¨[Info msg] Check class names {class_list}\n")

            

    ###################################### π μ μ²΄ μ΄λ―Έμ§ μ€μμ Dataframe νμ©νμ¬ class 19κ° κ°κ° 100κ°μ© λͺ¨μΌκΈ° π ##########################################
            
            ### κ° ν΄λμ€κ° 100κ°μ© λ€μ΄μλ μ΄λ―Έμ§ κ²½λ‘κ° λ΄κΈ΄ λ¦¬μ€νΈ μμ± ###
            print(f'β¨[Info msg] Start Collecting 100 images per class evenly \n')

            # λ¬΄μμ μΆμΆλ‘ ν΄λμ€ 19κ° 100κ°μ© μΆμΆ
            def make_cls_list(df,class_num):
                cls_list = random.sample(df.loc[df['class']==class_num, 'image'].tolist(), k=100)
                return cls_list

            # cls_list_1, 2, 3, 4, 5, 6 ,,,,,, 19
            print(f'β¨[Info msg] Generating each list consisting of 100 imgs per each class \n')
            for i in range(1,20): 
                globals()['cls_list_'+str(i)] = make_cls_list(df,i) # cls_list_1,2,3,...19 
            # len(cls_list_15) # 100


            ### cls_list_{}μ μλ μ΄λ―Έμ§ κ²½λ‘λ₯Ό 100_data/100_images/ (Mini_Dataset κ²½λ‘)μ μ μ₯ ###
            target_path = self.mini_root_dir
            tgt_img_path = os.path.join(target_path, '100_images')
            tgt_label_path = os.path.join(target_path, '100_labels')
            DATA_PATH = self.original_root_dir

            def movefile(filelist, movelist):
                for idx, filepath in enumerate(tqdm(filelist)):
                    copyfile(filepath, movelist[idx])
            
            # copyfile
            print(f'β¨[Info msg] Move images per class to Mini-Dataset Path \n')
            for i in range(1,20): # class 1 ~ 19

                # classλ³ listλ₯Ό νλμ© μΈμλ‘ μλ ₯ν΄μ λ¦¬μ€νΈ λ΄ μμλ€μ΄ νλμ© μ΄λν¨
                # Image data Movement
                print("[Image] Train class id : {}".format(i))
                movefile([os.path.join(filename) for filename in globals()['cls_list_'+str(i)]], [os.path.join(tgt_img_path,'Cls_'+ str(i) +'_'+ filename.split('/')[-1]) for filename in globals()['cls_list_'+str(i)]])

                # Label Movement
                print("[Label] Train class id : {}".format(i))
                movefile([os.path.join(DATA_PATH, 'label', filename.split('/')[-1].split('.')[0]+'.json') for filename in globals()['cls_list_'+str(i)]],\
                        [os.path.join(tgt_label_path, 'Cls_'+ str(i) +'_'+ filename.split('/')[-1].split('.')[0]+'.json') for filename in globals()['cls_list_'+str(i)]])


            ### ν΄λμ€κ° μ€λ³΅λ μ΄λ―Έμ§κ° μμ΄μ κ°μ κ°μκ° μΌμ΄λ¨. --> μ΄ 1,889κ° ###
            img_100_list = sorted(tqdm(glob(os.path.join(tgt_img_path,'*'))))
            label_100_list = sorted(tqdm(glob(os.path.join(tgt_label_path,'*'))))
            print(f'β¨[Info msg] Total Quantity of Mini-Dataset Data / Label --> {len(img_100_list)}/{len(label_100_list)} \n')
    

        
    ###################################### π Json (label νμΌ) μμ μλ class id μμ  π ##########################################
        '''
        < μλ³Έ json νμΌμ annotation νμ >
        ----->  {'data ID': 'S1', 'middle classification': '06', 'flags': {}, 'box': [1455, 776, 1772, 871], 'class': '13', 'large classification': '02'}
                {'data ID': 'S1', 'middle classification': '04', 'flags': {}, 'box': [1678, 728, 1740, 858], 'class': '10', 'large classification': '02'}

        < Class id μμ  ν >
        ---->   {'data ID': 'S1', 'middle classification': '06', 'flags': {}, 'box': [1455, 776, 1772, 871], 'class': 10, 'large classification': '02'}
                {'data ID': 'S1', 'middle classification': '04', 'flags': {}, 'box': [1678, 728, 1740, 858], 'class': 7, 'large classification': '02'}
  
        '''
        
        def class_id_change(self):

            target_path = self.mini_root_dir
            tgt_label_path = os.path.join(target_path, '100_labels')
            label_100_list = sorted(tqdm(glob(os.path.join(tgt_label_path,'*'))))
            print(f'β¨[Info msg] Change class ids in json file \n --> its location : {tgt_label_path} \n')

            # Class id 2,3,4 --> 1  //  Class id 5 ~ 23 --> κ°κ° -3 
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

                    # κΈ°μ‘΄ json νμΌ λ?μ΄μ°κΈ°
                    with open(file_path[i], 'w', encoding='utf-8') as file:
                        json.dump(jsonFile,file, indent='\t')

            class_change_json(label_100_list)
            print(f'β¨[Info msg] Change class ids in json file Complete \n')



    ###################################### π 100_data/100_image (Mini-dataset κ²½λ‘) μμ κ° ν΄λμ€λ³ μΈλ±μ€λ₯Ό μΆμΆν΄μ κ·Έκ²μ μ΄μ©ν΄μ λ°μ΄ν° λΆν  π ##########################################
        def extract_indexes_per_class(self):


            # κ° ν΄λμ€λ³ 0.15 ratioλ‘ Train / Val Split
            print(f'β¨[Info msg] Split data into Train and Valid')

            ### νμΌλͺ μΆμΆ ###
            print(f'β¨[Info msg] Extract file_names \n')
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


            ### class λ³ μ΄λ―Έμ§ κ°μ νμΈ
            print(f'β¨[Info msg] Check Quantity of Images per Class \n')


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
            print(f'β¨[Info msg] Check Data Quanitty per class \n {numofCat}')
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

            ### Validation Ratio μ€μ κ³Ό κ·Έμ λ°λ₯Έ validation data μλ μΆμΆ ###
            val_ratio = [int(numofCat.get(i)* self.val_ratio) for i in range(1,(self.class_num+1))]
            print(f'β¨[Info msg] Validation ratio : {val_ratio} and Validation Data Quantity along this ratio ---> \n {val_ratio} \n length : {len(val_ratio)}')


            ### Class λ³ λ¦¬μ€νΈ λ΄ μΈλ±μ€ μΆμΆ
            # --> imgs κΈ°μ€ κ° ν΄λμ€λ³ μΈλ±μ€ μμ μμΉ
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


            ### Train, Validμ© μΈλ±μ€ μμ±ν΄μ class λ³ κ· λ±ν λΉμ¨λ‘ Data Split μν ###

            # split the 100_data into two groups
            # trian 0.85, val 0.15
            # imgs, labels


            # val_ratio : 0.15
            def making_each_class_idx(total_list, class_num, indexes_dict, val_ratio_list,img_quantity_dict,mode):
                '''
                < args >
                total_list : class λ³λ‘ λ°μ΄ν°λ₯Ό λͺ¨μλμ λ°μ΄ν°μμ λ΄μ list
                class_num : μΈλ±μ€λ₯Ό μμ±ν  ν΄λμ€ μ«μ (int)
                indexes_dict : imgs κΈ°μ€ κ° ν΄λμ€λ³ μΈλ±μ€ μμ μμΉκ° λ΄κ²¨ μλ dict
                val_ratio_list : ν΄λμ€λ³ λ°μ΄ν°μ val_ratioλ₯Ό κ³±ν΄μ λμ¨ μλμ΄ λ΄κ²¨ μλ list
                img_quantity_dict : ν΄λμ€λ³ λ°μ΄ν° μλμ΄ λ΄κ²¨ μλ dict
                mode : train or valid 
                '''

                train_idx = []
                valid_idx = []
                print(f'Total Index length :: {len(total_list)}')
                print(f'Total number of images of class <{class_num}> is {img_quantity_dict.get(class_num)}')
                key_dict = {x:x for x in range(1,20)}


                if mode == 'valid':
                    
                    # classκ° μΌμΉν  λ

                    if int(class_num) == int(key_dict.get(class_num)):
                        print(f'class <<{class_num}>> appending into VALID_list... ')
                        
                        # class start_index : class_start_index + val_ratio
                        for i in tqdm(range(indexes_dict.get(class_num), indexes_dict.get(class_num)+val_ratio_list[class_num-1],1), desc='Making β Valid β Indexes....', leave=True): 
                            valid_idx.append(i)
                                
                        print('\n Valid Index')
                        print(f'Idx :: Start_index <{indexes_dict.get(class_num)}> --> End_index <{(indexes_dict.get(class_num)+val_ratio_list[class_num-1]-1)}>')
                        print(f'Class <{class_num}> Train Start Index : <{indexes_dict.get(class_num) + val_ratio_list[class_num-1]}>')
                        print(f'length of valid list : {len(valid_idx)}')

                        return valid_idx


                    # ν΄λμ€ 19μ λ€μ ν΄λμ€λ μμΌλ―λ‘ λ³λ μ²λ¦¬
                    elif int(class_num) == int(key_dict.get(19)):

                        #  898  ~ 898 * 0,15 κΉμ§
                        for i in tqdm(range(indexes_dict.get(class_num) ,indexes_dict.get(class_num) + val_ratio_list[class_num-1]),indexes_dict.get(1), leave=True):

                            valid_idx.append(i)

                        print('Valid Index')
                        print(f'Idx :: Start_index <{indexes_dict.get(class_num)}> --> End_index <{indexes_dict.get(class_num)+val_ratio_list[class_num-1]-1}>')
                        print('Next Class Index :  NONE')
                        print(f'length of valid list : {len(valid_idx)}')
                        print('DOOONNNNNEEEEEEE!!!!!', '\n')
                                
                        return valid_idx


                if mode == 'train' : 

                    # TypeError: 'int' object is not subscriptableλ μΈλ±μ€λ₯Ό κ°μ§μλ κ°μ μΈλ±μ€λ₯Ό κ°μ§κ² μ½λλ₯Ό μ§€ κ²½μ° λ°μνλ μ€λ₯μ΄λ€.
                    # classκ° μΌμΉν  λ
                    if int(class_num) == int(key_dict.get(class_num)):
                        print(f'class <<{class_num}>> appending into TRAIN_list... ')

                        # class = 19μΌ λλ
                        if int(class_num) == int(key_dict.get(19)):

                            #  (valid_idx+1) ~ 996 (class 1 idx-1)κΉμ§
                            for i in tqdm(range(indexes_dict.get(class_num) + val_ratio_list[(class_num-1)], indexes_dict.get(1)), leave=True):
                                train_idx.append(i)

                            print('Train Index')
                            print(f'Idx :: Start_Index <{indexes_dict.get(class_num)+val_ratio_list[class_num-1]}> --> End_Index <{(indexes_dict.get(1)-1)}>')
                            print('Next Class Index :  NONE')
                            print(f'length of train list : {len(train_idx)}')
                            print('DOOONNNNNEEEEEEE!!!!!', '\n')

                            return train_idx


                        # class 10μ μμ μΈλ±μ€κ° 0μ΄λ―λ‘ class = 9μΌ λλ λ³λ μ²λ¦¬       
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

                        # μ μμΌ λλ    
                        # class_start_index + val_ratio : next_class_index-1
                        for i in tqdm(range(indexes_dict.get(class_num) + val_ratio_list[class_num-1] , indexes_dict.get(class_num+1)), desc = 'Making β Train β Indexes.....', leave=True):
                            train_idx.append(i)
                            
                        print('\n Train Index')
                        print(f'Idx :: Start_Index <{indexes_dict.get(class_num)+val_ratio_list[class_num-1]}> --> End_Index <{(indexes_dict.get(class_num+1)-1)}>')
                        print(f"Next Class Index : <{indexes_dict[class_num+1]}>")
                        print(f'length of train list : {len(train_idx)}')
                            
                        return train_idx

                

            # ν΄λμ€ λ³ μΈλ±μ€λ₯Ό λ§λ€μ΄μ νλμ μΈλ±μ€μ μ΅μ’μ μΌλ‘ μμ±
            def Total_Index_Making(quantity):
                '''
                < args > 
                quantity : indexλ₯Ό λ§λ€ ν΄λμ€μ (λ μ«μ + 1)  ex : 20 --> 1 ~ 19 
                '''

                print(f'Making TOTAL INDEX LIST OF NUMBER OF <{(quantity-1)}> CLASSES \n')
                total_tr_img_idxes = []
                total_tr_label_idxes = []
                total_val_img_idxes = []
                total_val_label_idxes = []

                # class numμ λ°μμ class λ³ μΈλ±μ€ λ§λλ ν¨μμ μΈμλ‘ μλ ₯
                for class_num in range(1,quantity,1):

                    val_img_idx = making_each_class_idx(imgs,class_num,indexes, val_ratio,numofCat,mode='valid')
                    val_label_idx = making_each_class_idx(labels,class_num,indexes, val_ratio,numofCat,mode='valid')
                    tr_img_idx = making_each_class_idx(imgs,class_num,indexes, val_ratio,numofCat, mode='train')
                    tr_label_idx = making_each_class_idx(labels,class_num,indexes, val_ratio,numofCat, mode='train')

                    total_tr_img_idxes.append(tr_img_idx)
                    total_tr_label_idxes.append(tr_label_idx)
                    total_val_img_idxes.append(val_img_idx)
                    total_val_label_idxes.append(val_label_idx)

                    print('ββββββββββββββββββββββββββββββββββββββββββββββββββββ')
                    print(f'CHECK WHETHER IMG LIST AND LABEL LIST OF CLASS <<{class_num}>> ARE MATCHED')
                    print('Train INDEXES of Imgs and Labels are matched ? :: <<{}>> '.format(sorted(tr_img_idx)==sorted(tr_label_idx)))
                    print('Valid INDEXES of Imgs and Labels are matched ? :: <<{}>> '.format(sorted(val_img_idx)==sorted(val_label_idx)))
                    print('ββββββββββββββββββββββββββββββββββββββββββββββββββββ \n\n\n')

                return total_tr_img_idxes, total_tr_label_idxes, total_val_img_idxes, total_val_label_idxes

            # class 1 ~ 19 train imgs, labels / valid imgs, labelsμ μΈλ±μ€λ€μ΄ κ°κ° μ μ₯λ¨.    
            tr_img_idx, tr_label_idx , val_img_idx, val_label_idx = Total_Index_Making((self.num_class+1))

            Images = sorted(name_parsing(img_100_list))
            Labels = sorted(name_parsing(label_100_list))
        

    ###################################### π Train, Validμ© μΈλ±μ€ νμ©ν΄μ Annotation μ© Meta file (txtνμΌ) μμ± π ##########################################
        def gen_annotation(self):

            print(f'\n β¨[Info msg] Start Generating Meta file (.txt) for Annotation using the indexes \n')

            def flatten_list(target_list):
                target_list = itertools.chain(*target_list)
                target_list_ = list(target_list)

                return target_list_

            # 2μ°¨μ listλ‘ λμ΄ μμ΄μ 1μ°¨μμΌλ‘ λ³κ²½
            tr_img_idx = flatten_list(tr_img_idx)
            tr_label_idx = flatten_list(tr_label_idx)
            val_img_idx = flatten_list(val_img_idx)
            val_label_idx = flatten_list(val_label_idx)

            global Images, Labels
            print(f'β¨[Info msg] Check Quantity of the Indexes')
            print(f'Image Difference (Should be 0) : {len(Images) - len(tr_img_idx) - len(val_img_idx)}')
            print(f'Label Difference (Should be 0) : {len(Labels) - len(tr_label_idx) - len(val_label_idx)}')

            print(f'Train Data Quantity : {len(tr_img_idx)}')
            print(f'Train Label Quantity : {len(tr_label_idx)}')
            print(f'Valid Data Quantity : {len(val_img_idx)}')
            print(f'Valid Label Quantity : {len(val_label_idx)}')


            # 1μ°¨μμΌλ‘ λ λ¦¬μ€νΈ μμ μμλ₯Ό μΆμΆνμ¬ μ΅μ’ μμκ° λ΄κΈ΄ λ¦¬μ€νΈ μμ±
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

            print('βββ Check whether they are matched..... βββ')
            print(f'train : {tr_imgs_list == tr_labels_list}')
            print(f'valid : {val_imgs_list == val_labels_list}\n\n')


            ### ann_file (meta_file) generation ###
            print(f'β¨[Info msg] Annotation file (meta_file) Generation')
            os.system('cd /content/drive/MyDrive/airplane_custom')
            tr_df = pd.DataFrame({'tr_filename' : tr_imgs_list})
            val_df = pd.DataFrame({'val_filename': val_imgs_list, 'val_labels_name' : val_labels_list})

            # # κΈ°μ‘΄ μΈλ±μ€ μ­μ  
            # # Cross Validation μ­ν 
            # os.system('rm ./100_data/train.txt')
            # os.system('rm ./100_data/val.txt')

            tr_df['tr_filename'].to_csv(f'{self.mini_root_dir}/train.txt', index=False, header=False)
            val_df['val_filename'].to_csv(f'{self.mini_root_dir}/val.txt', index=False, header=False)
            print(f"β¨[Info msg] Saving files....to {f'{self.mini_root_dir}/'}")

            image_tlist = mmcv.list_from_file(f'{self.mini_root_dir}/train.txt')
            image_vlist = mmcv.list_from_file(f'{self.mini_root_dir}/val.txt')

            print(f'Length of total tr_imgs names : {len(image_tlist[:])}')
            print(f'tr_imgs names : \n {image_tlist[:5]}\n')
            print(f'Length of total val_imgs names : {len(image_vlist[:])}')
            print(f'val_imgs names : \n {image_vlist[:5]}') 

            print(f"\n\n β¨[Info msg] Entire Process has done :)")

    # μ€ν
    mdmaker = MDMaker()
    mdmaker.class_name_change()
    mdmaker.class_id_change()
    mdmaker.extract_indexes_per_class()
    mdmaker.gen_annotation()


if __name__ == '__main__':
    main()
