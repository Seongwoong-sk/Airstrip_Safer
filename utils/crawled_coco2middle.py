
# ๐ Roboflow์์ ํฌ๋กค๋งํด์ ์ป์ ์ด๋ฏธ์ง๋ค๋ก ๋ฐ์ดํฐ์์ ์์ฑํ์์. 
# ๐ ์์ฑ๋ ๋ฐ์ดํฐ์์ Label ํ์์ "COCO Format" ํ์์ผ๋ก ๋ผ์์ด์ MMDetection์ "Middle Format"์ผ๋ก ๋ณํํด์ Original Dataset์ผ๋ก ์ฎ๊ธฐ๋ ํ์ด์ฌ ํ์ผ
# ๐ AI HUB์ ์๋ณธ ๋ฐ์ดํฐ์์ Label ํ์์ Pascal VOC ์ฒ๋ผ ๊ฐ ์ด๋ฏธ์ง๋ง๋ค json ํ์ผ์ด ํ์ฑ๋ผ์์.

# 1๏ธโฃ Class 4 (Road Surface disorder)์ Class 5 (Obstacle)์ FOD๋ก ๋ณํฉํ  ์ ์์ด์ ํด๋์ค FOD๋ก ๋ณํฉ
# --> ์ ์ฒด ๋ฐ์ดํฐ์ json ํ์ผ์ ์ ๊ทผํด์ ํด๋์ค id ๋ณ๊ฒฝ

# 2๏ธโฃ ํฌ๋กค๋งํ ๋ฐ์ดํฐ ์ ์ฒ๋ฆฌ ํ ์ถ๊ฐ
# --> ์๋ณธ json ํ์ผ ๋ด์ ๊ธฐ๋ก๋์ด ์๋ image id๋ฅผ ์ด์ฉํ์ฌ img & json ํ์ผ๋ช ๋ณ๊ฒฝ
# --> ๊ณ ์ ๋ฒํธ ์ด์ฉํด์ coco json ํ์ผ ํ ๊ฐ์์ ann img๋ง๋ค ๋์๋๋ json ํ์ผ ์์ฑ
# --> ์๋ณธ json ํ์ผ ๋ด์ ํ๋ จ์ ๋ถํ์ํ ๋ฐ์ดํฐ๋ค์ ์๋กญ๊ฒ ๋ง๋  json ํ์ผ ๋ด์ ์ถ๊ฐ ์ํค์ง ์์

'''
< Structure Example >
โโโ airplane_custom (Root_Dir)
โ   โโโ Addition (Crawled Dir)
โ         โโโ fire_truck (classes)
โ               โโโ train
โ               โโโ train_label
โ               โโโ valid
โ               โโโ valid_label
โ               โโโ (test)
โ               โโโ (test_label)
โ               โโโ train_annotations_coco.json
โ               โโโ valid_annotations_coco.json
โ               โโโ (test_annotations_coco.json)
โ         โโโ special_vehicle
โ               โโโ ""
โ         โโโ step_car
โ               โโโ ""
โ         โโโ road_sweeper
โ               โโโ ""
โ         โโโ weed_removal
โ               โโโ ""
โ
โ   โโโ 21000_Dataset (Original Dataset)
โ         โโโ train_data
โ         โโโ train_label
โ         โโโ valid_data
โ         โโโ valid_label
โ         โโโ test_data
โ         โโโ test_label
โ         โโโ train.txt (Annotation file)
โ         โโโ val.txt (Annotation file)
โ         โโโ test.txt (Annotation file)

'''

###################################### ๐ Libraries ๐ ################################################################

import os
import cv2
import json
from glob import glob
from tqdm import tqdm
from shutil import copyfile

def main():

    class Conversion:

        def __init__(self):

            # Original Dataset Root dir
            self.original_root_dir = f'/content/drive/MyDrive/airplane_custom/21000_Dataset'

            # Crawled Dataset Folder Root dir
            self.crawled_root_dir = '/content/drive/MyDrive/airplane_custom/Addition'


    ###################################### ๐ ํด๋์ค ๋ณํฉ ๐ ################################################################

        def class_conversion(self):

            print(f"[Info msg] โจ Class Combination Start 4 + 5 --> 4 \n")
            ## Json์ Class 4 (Road surface disorder) + Class 5 (Obstacle) ๋ณํฉ
            ## ---> Class 4๋ FOD๋ก ๋ณ๊ฒฝ
            ## ์ต์ข 19๊ฐ

            CLASSES = [ 
                'Aircraft',  
                'Rotorcraft', 
                'Road surface facility',  
                'Obstacle (FOD)', 
                'Bird', 
                'Mammals', 
                'Worker', 
                'Box', 
                'Pallet', 
                'Toyinka', 
                'Ramp bus', 
                'Step car', 
                'Fire truck',
                'Road sweeper', 
                'Weeding vehicle', 
                'Special vehicle', 
                'Forklift', 
                'Cargo loader', 
                'Tug Car'] 


            # class_num : class_name dict ์์ฑ
            global dict_ # coco2middle ํจ์์์ ์ฌ์ฉํ๊ธฐ ๋๋ฌธ์ ์ ์ญ ๋ณ์๋ก ์ ์ธ
            dict_ = {i+1:k for i,k in enumerate(CLASSES)}

            # ํด๋์ค๊ฐ 1~20์ผ๋ก ์ด๋ฃจ์ด์ ธ์์ด์ 5~20์ -1์ฉ ํ๋ฉด 1~19๋ก ๋จ.
            def class_change(label_file,change_start_idx, change_end_idx , mode=None):

                print(f'--- Start Changing class from {change_start_idx} to {change_end_idx}  ---')

                cnt=0
                change_range_ = [i for i in range(change_start_idx, change_end_idx+1)]
                change_cnt=0

                for i in tqdm(range(len(label_file)), desc= f'Changing Class in <{mode}> json files'): 

                    with open(label_file[i], 'r') as f:
                        jsonFile = json.load(f)

                        # annotations์ ์ ๊ทผํด์ class๋ฅผ -1์ฉ ๋ง๋ฆ.
                        for ann_data in jsonFile['annotations']:
                            ann_data['class'] = int(ann_data['class'])
                            if ann_data['class'] in change_range_:
                                ann_data['class'] = int(ann_data['class'])-1 # -1
                                cnt+=1
                                
                    with open(label_file[i], 'w', encoding='utf-8') as file_:
                        json.dump(jsonFile, file_, indent='\t')
                        # print('json ํ์ผ ์์  ์๋ฃ')
                        change_cnt +=1
                
                print(f'{mode}_Class๊ฐ ๋ฐ๋ ํ์ : {cnt} \n {mode}_Json์ด ๋ฐ๋ ํ์ : {change_cnt}')


            # ํด๋น ํด๋์ค id๊ฐ json ํ์ผ์ ์๋์ง ํ์ธํ๋ ํจ์
            def class_checking(label_file, class_to_check, mode=None):

                print(f'--- Start Checking whether class {class_to_check} exists or not ---')

                cnt=0
                list_ = []

                for i in tqdm(range(len(label_file)), desc= f'Checking class {class_to_check} in <{mode}> json files'): 

                    with open(label_file[i], 'r') as f:
                        jsonFile = json.load(f)

                        for ann_data in jsonFile['annotations']:
                            ann_data['class'] = int(ann_data['class'])
                            if ann_data['class'] in [class_to_check]:
                                list_.append(jsonFile['image']['filename'])
                                cnt+=1
                
                print(f'{mode} Class ({class_to_check}) ๊ฐ์ : {cnt}')

                return list_

            
            ### Data Checking ###
            print(f"[Info msg] โจ Check Quantity of Original Dataset \n ---> Original Data Path : {self.original_root_dir}/train or valid or test_data or label/")
            tr_imgs = sorted(list(glob(os.path.join(f'{self.original_root_dir}/train_data/*'))))
            tr_labels = sorted(list(glob(os.path.join(f'{self.original_root_dir}/train_label/*'))))

            val_imgs = sorted(list(glob(os.path.join(f'{self.original_root_dir}/valid_data/*'))))
            val_labels = sorted(list(glob(os.path.join(f'{self.original_root_dir}/valid_label/*'))))

            test_imgs = sorted(list(glob(os.path.join(f'{self.original_root_dir}/test_data/*'))))
            test_labels = sorted(list(glob(os.path.join(f'{self.original_root_dir}/test_label/*'))))

            print(f"Quantity of Train Data : {len(tr_imgs)} / Quantity of Train Label : {len(tr_labels)}") 
            print(f"Quantity of Valid Data : {len(val_imgs)} / Quantity of Valid Label : {len(val_labels)}")
            print(f"Quantity of Test DAta : {len(test_imgs)} / Quantity of Test Label : {len(test_labels)} \n")



            ### class 5๋ถํฐ 20๋ง -1์ฉํ๋ฉด 1,2,3,4,5,6,.....,19๋ก ์์ฑ ###
            print(f"[Info msg] โจ Change Train class 5 ~ 20 into 4 ~ 19 in json files \n")
            class_change(tr_labels,5,20,'Train')
            print(f"[Info msg] โจ Change Valid class 5 ~ 20 into 4 ~ 19 in json files\n")
            class_change(val_labels,5,20,'Valid')
            print(f"[Info msg] โจ Change Test class 5 ~ 20 into 4 ~ 19 in json files\n")
            class_change(test_labels,5,20,'Test')

            print(f"[Info msg] โจ  Checking whether class 20 exists or not in json files(20 Must not exist) \n")
            tr_checking_20 = class_checking(tr_labels,20,'Train')
            val_checking_20 = class_checking(val_labels,20,'Valid')
            test_checking_20 = class_checking(test_labels,20,'Test')
            


    ############################# ๐ ํฌ๋กค๋งํ ๋ฐ์ดํฐ ์ ์ฒ๋ฆฌ ํ ์ค๋ฆฌ์ง๋ ๋ฐ์ดํฐ์์ ์ถ๊ฐ ๐ ###################################

        def crawled_preprocess_format_conversion(self):

            print(f"[Info msg] โจ Start Adding Crawled Data to the Original Dataset \n")


            ### Loading & Checking Quantity of Crawled Data into lists ###
            print(f"[Info msg] โจ Loading & Checking Quantity of Crawled Data into lists \n")
            print(f"[Info msg] โจ Train Data")
            t_fire = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/fire_truck/train/*'))))
            t_special = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/special_vehicle/train/*'))))
            t_step = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/step_car/train/*'))))
            t_road = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/road_sweeper/train/*'))))
            t_weed = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/weed_removal/train/*'))))

            print('Train : ', len(t_fire), len(t_special), len(t_step), len(t_road), len(t_weed))
            print(f"Train Sum : {sum([len(t_fire), len(t_special), len(t_step), len(t_road), len(t_weed)])}")

            print(f"\n [Info msg] โจ Valid Data")
            v_fire = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/fire_truck/valid/*'))))
            v_special = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/special_vehicle/valid/*'))))
            v_step = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/step_car/valid/*'))))
            v_road = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/road_sweeper/valid/*'))))
            v_weed = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/weed_removal/valid/*'))))

            print('Valid : ', len(v_fire), len(v_special), len(v_step), len(v_road), len(v_weed))
            print(f"Valid Sum : {sum([len(v_fire), len(v_special), len(v_step), len(v_road), len(v_weed)])}")



            ### Roboflow Coco ํ์์ธ json ํ์ผ ๋ด์ ๊ธฐ๋ก๋์ด ์๋ image id๋ฅผ ์ด์ฉํ์ฌ img & json ํ์ผ๋ช ๋ณ๊ฒฝ ###
            '''
            << Roboflow์์ ์ถ์ถํ COCO ํ์์ json ํ์ผ >> 

            {
            'info': {'year': '2022',
            'version': '3',
            'description': 'Exported from roboflow.ai',
            'contributor': '',
            'url': 'https://public.roboflow.ai/object-detection/undefined',
            'date_created': '2022-09-05T12:47:53+00:00'},
            'licenses': [
            {'id': 1,
            'url': 'https://creativecommons.org/licenses/by/4.0/',
            'name': 'CC BY 4.0'}
            ],

            'categories': [{'id': 0, 'name': 'Step-car', 'supercategory': 'none'},
            {'id': 1, 'name': 'Step car', 'supercategory': 'Step-car'}],

            'images': [
            {'id': 0, ๐ ์ด๊ฑธ ํ์ฉ ๐ 
            'license': 1,
            'file_name': 'airport-step-car_173_jpg.rf.3d8188ea5df2295147ff0e09e8adbc30.jpg',
            'height': 480,
            'width': 640,
            'date_captured': '2022-09-05T12:47:53+00:00'},
            {'id': 1, ๐ ์ด๊ฑธ ํ์ฉ ๐
            'license': 1,
            'file_name': 'airport-step-car_6_jpg.rf.566299ef20aa0ec399bf14d3c087bcd0.jpg',
            'height': 267,
            'width': 400,
            'date_captured': '2022-09-05T12:47:53+00:00'},
            ]

            'annotations': [
            {'id': 0,
            'image_id': 0, ๐ ์ด๊ฑธ ํ์ฉ ๐
            'category_id': 1,
            'bbox': [149, 21, 354.59, 446.36],
            'area': 158274.7924,
            'segmentation': [],
            'iscrowd': 0},
            {'id': 1,
            'image_id': 1, ๐ ์ด๊ฑธ ํ์ฉ ๐
            'category_id': 1,
            'bbox': [57, 16, 289.24, 214.78],
            'area': 62122.9672,
            'segmentation': [],
            'iscrowd': 0},
            ]
            }
            '''

            # json ๋ด image id์ ์ ๊ทผํด์ id๊ฐ ํฌํจ๋ ํ์ผ๋ช์ผ๋ก ๋ณ๊ฒฝ ํ ํฌ๋กค๋ง ํด๋ ๊ฒฝ๋ก์ ์ ์ฅํ๋ ํจ์
            def change_filename_save(json_file, data_root, target_root, mode):

                cnt = 0
                change = 0
                finish = 0

                with open(json_file,'r') as f:

                    jsonFile = json.load(f)

                    # images ํค์ ์ ๊ทผ
                    for img_data in tqdm(jsonFile['images']):

                        # images ํค์ ์๋ ๊ฐ ์ด๋ฏธ์ง์ id๋ฅผ ์ถ์ถ
                        img_id = int(img_data['id'])
                        
                        file_name = img_data['file_name']
                        file_name = file_name[:file_name.rfind('.')]

                        # annotations ํค์ ์๋ ์ด๋ฏธ์ง id๋ฅผ ์ถ์ถ (images ํค์ ์๋ id๋ ๋์ผ)
                        for anno_data in jsonFile['annotations']:

                            if img_id == anno_data['image_id']:
                                print(f'\n\n โโ image_id = {img_id} ์ด๋ฏธ์ง โโ')
                                print(f"Original filename : {img_data['file_name']} //// Original img_id : {img_id} ")

                                print(' json ๋ด ํ์ผ๋ช ๋ณ๊ฒฝ!!!!!  ')

                                # ์ค์  ๋ฐ๊ฟ ์ด๋ฆ๋ช. ์ด๋ฆ ๋ณ๊ฒฝ
                                img_data['file_name'] = f'{mode}_{target_root}_{str(img_id)}.jpg'
                                print(f'--> json ๋ด ํ์ผ๋ช ๋ณ๊ฒฝ ์๋ฃ!!  :::', img_data['file_name'], '\n')
                                cnt+=1

                                print('--> Json ๋ด ๋ณ๊ฒฝ๋ ํ์ผ๋ช์ผ๋ก ์ค์  ์ด๋ฏธ์ง ์ด๋ฆ ๋ณ๊ฒฝ!!')
                                

                                if not os.path.exists(os.path.join(data_root,target_root,mode,img_data['file_name'])):
                                    # ์ด๋ฏธ์ง ๋ณ๊ฒฝํด์ ์ ํด์ง ๊ฒฝ๋ก์ ์ ์ฅ
                                    os.rename(os.path.join(data_root, target_root, mode, file_name+'.jpg'),
                                            os.path.join(data_root, target_root, mode, img_data['file_name']))
                                    
                                    print(f" ๋ณ๊ฒฝ ์๋ฃ!!!\n--> ๋ณ๊ฒฝ๋ ์ด๋ฏธ์ง ๊ฒฝ๋ก : {os.path.join(data_root, target_root, mode, img_data['file_name'])}")
                                    change+=1


                # ๊ธฐ์กด json ํ์ผ ๋ฎ์ด์ฐ๊ธฐ
                # annotation์ด ์๋ ์ด๋ฏธ์ง๋ง ์ถ๋ ค์ ํ์ผ๋ช ๋ณ๊ฒฝํ์.
                with open(json_file, 'w', encoding='utf-8') as file_:
                    json.dump(jsonFile, file_, indent="\t")
                    print('json ํ์ผ ์ต์ข ์์  ์๋ฃ!!')
                    finish +=1
                
                print(f"json ๋ด ํ์ผ๋ช ๋ณ๊ฒฝ ๋ณ๊ฒฝ๋ ํ์ : {cnt} // ์ด๋ฏธ์ง ID์ผ๋ก ํ์ผ๋ช ๋ณ๊ฒฝ๋ ํ์ : {change} // ์ต์ข ์์ ๋ ํ์ผ ๊ฐฏ์ : {finish}")


            DATA_ROOT = f'{self.crawled_root_dir}'
            TARGET_ROOT = 'fire_truck' # road_sweeper, special_vehicle, step_car, fire_truck, weed_removal
            MODE = 'valid' # train or valid

            print(f"[Info msg] โจ Start Changing file_names in jsons and save to crawled folder \n")
            change_filename_save(json_file=f'{DATA_ROOT}/{TARGET_ROOT}/{MODE}_annotations.coco.json', 
                        data_root = DATA_ROOT,
                        target_root=TARGET_ROOT, 
                        mode=MODE)


            # Image ID (๊ณ ์ ๋ฒํธ) ์ด์ฉํด์ coco json ํ์ผ ํ ๊ฐ์์ ann img๋ง๋ค ๋์๋๋ json ํ์ผ ์์ฑ

            tr_img_lists = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/fire_truck/train/*'))))
            val_img_lists = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/fire_truck/valid/*')))) 

            tr_truc_elems = os.listdir(os.path.join(f'{self.crawled_root_dir}/weed_removal/train'))
            val_truc_elems = os.listdir(os.path.join(f'{self.crawled_root_dir}/weed_removal/valid'))

            print(val_truc_elems[1]) # valid_weed_removal_1.jpg
            print(val_truc_elems[1].split('_')[-1].split('.')[0]) # 1



            # ์ด๋ฏธ์ง๋ช์์ ๊ณ ์  id ๊บผ๋ด์ ๋ฆฌ์คํธ์ ์ ์ฅํ๋ ํจ์
            def extracting_eigen_num(data_root,target,tr_mode='train',val_mode='valid'):
                '''
                < args >
                data_root : ํฌ๋กค๋งํ ๋ฐ์ดํฐ๋ค์ ํด๋๋ณ๋ก ๊ตฌ๋ถํด๋์ ๊ฒฝ๋ก
                target : ํฌ๋กค๋งํ ์ด๋ฏธ์ง๋ค์ ๋ผ๋ฒจ ์ด๋ฆ (ex : step_car, weed_removal)
                tr_mode : Train Data Folder Accessing
                val_mode : Valid Data Folder Accessing
                '''

                tr_truc_elems = os.listdir(os.path.join(f'{data_root}/{target}/{tr_mode}'))
                val_truc_elems = os.listdir(os.path.join(f'{data_root}/{target}/{val_mode}'))

                # ์ด๋ฏธ์ง๋ช์์ ๊ณ ์  id ๊บผ๋ด๊ธฐ

                tr_unique_num = []
                val_unique_num = []

                for i in range(len(tr_truc_elems)):

                    num = int(tr_truc_elems[i].split('_')[-1].split('.')[0]) # ex : valid_weed_removal_1.jpg --> 1
                    tr_unique_num.append(num)
                
                for k in range(len(val_truc_elems)):

                    val_num = int(val_truc_elems[k].split('_')[-1].split('.')[0])
                    val_unique_num.append(val_num)

                print(f"โจ[Info msg] Unique ID of Train / Valid Images : {len(tr_unique_num)} / {len(val_unique_num)}")

                return tr_unique_num, val_unique_num


            # ์ด๋ฏธ์ง์ ๊ฐ์ฒด์ ๋ํ ์ ๋ณด๊ฐ ํ๋์ json ํ์ผ์ ๋ค ๋ค์ด์๋ COCO ํ์ json์ ์ ๊ทผํ์ฌ 
            # Image ID๋ฅผ ์ด์ฉํด ํ ์ด๋ฏธ์ง์ ๋์ํ๋ ํ๋์ json ํ์ผ๋ค์ ์์ฑํ๋ ํจ์
            def coco2middle(open_json, unique_list, class_of_this_object,json_path_to_save,mode):

                '''
                <args> 
                open_json : ์๋ณธ coco json ํ์ผ
                unique_list : annotation์ด ์๋ ์ด๋ฏธ์ง์ ํ์ผ๋ช์ ์๋ num๋ค์ด ๋ชจ์ฌ์ ธ์๋ list
                class_of_this_object : ์ง์ ํ  ํด๋์ค
                json_path_to_save : json files์ ์ ์ฅํ  ๊ฒฝ๋ก
                mode : train or valid
                '''

                with open(open_json,'r') as f:
                    ann_data = json.load(f)
                    cnt=0

                    for img_data in ann_data['images']:


                        img_id = int(img_data['id'])
                        file_num = int(img_data['file_name'].split('_')[-1].split('.')[0]) # ํ์ผ๋ช์ ์๋ ์ด๋ฏธ์ง ๋ฒํธ
                        
                        for anno_data in tqdm(ann_data['annotations']):

                            if (img_id == anno_data['image_id']) and (file_num in unique_list):

                                print(f'{mode} file_num : {file_num} ์ฐจ๋ก์๋๋ค // img_id : {img_id}')
                                cnt+=1

                            # ๋ณ๊ฒฝ๋ ํ์ผ๋ช์ ๋์ํ๋ json ํ์ผ์ ๋ง๋ค๊ณ  ๊ทธ ํ์ผ์ ๋ง๋ ann ์ ๋ณด ์๋ ฅํด์ json ์๋ก ์์ฑ

                                file_name = img_data['file_name'].split('.')[0]
                                if not os.path.exists(json_path_to_save): os.mkdir(json_path_to_save)
                                label_path = json_path_to_save


                                with open(os.path.join(label_path,file_name+'.json'),'w') as f_out:


                                # json ํ์ผ ๋ด์ฉ ๋ฃ์ ๊ณณ

                                    data_dict = {} # base format

                                    data_dict['image'] = {}  # base format์ ์ด๋ฏธ์ง 1์ฅ  :: dict
                                    data_dict['annotations'] = [] # ์ด๋ฏธ์ง 1์ฅ์ ์๋ ์ฌ๋ฌ ๊ฐ์ ๊ฐ์ฒด :: list
                                    data_dict['image']['resolution']= [0 for _ in range(2)]  # resolution ๋ด์ 2 ํฌ๊ธฐ์ ๋ฆฌ์คํธ ์์ฑ

                                    data_dict['image']['filename'] = img_data['file_name']
                                    data_dict['image']['resolution'][0]= img_data['width']
                                    data_dict['image']['resolution'][1] = img_data['height']

                                    base_format = {} # annotations list์์ ๋ฃ์ ์ฌ๋ฌ ๊ฐ์ dic
                                    base_format['class'] = class_of_this_object

                                    # coco ํํ [x1,y1,bbox_w,bbox_h] ๋ฅผ ํ๋ก์ ํธ bbox ํ์์ธ [x1,y1,x2,y2]๋ก ๋ณ๊ฒฝ
                                    # BBox ํํ๋ฅผ ๊ธฐ์กด COCOํํ [x1,y1,bbox_w,bbox_h]๋ก ๋ฃ์ด์ ๋ณํํ json ์ ๊ทผํด์ bbox ๊ฐ ์์ 
                                    base_format['box'] = [0 for _ in range(4)] # # bbox ๋ด์ 4 ํฌ๊ธฐ์ ๋ฆฌ์คํธ ์์ฑ
                                    base_format['box'][0] = anno_data['bbox'][0] # x1
                                    base_format['box'][1] = anno_data['bbox'][1] # y1
                                    base_format['box'][2] = anno_data['bbox'][0] + anno_data['bbox'][2]# x1 + bbox_w
                                    base_format['box'][3] = anno_data['bbox'][1] + anno_data['bbox'][3]# x1 + bbox_w

                                    data_dict['annotations'].append(base_format)
                                                
                                    json.dump(data_dict , f_out, indent='\t') # img๋ช์ ๊ฐ์ง json ํ์ผ ๊ฐ๊ฐ ์์ฑ
                    
                    print(f"{cnt}๊ฐ ์ฑ๊ณต!!")


            data_root = f'{self.crawled_root_dir}'
            target = 'weed_removal' #road_sweeper, special_vehicle, step_car, fire_truck, weed_removal
            mode = 'valid' # train
            class_ = 'Weeding vehicle' # 'Road sweeper', 'Special vehicle', 'Step car', 'Weeding vehicle', 'Fire truck'


            print(f"[Info msg] โจ Extracting each eigen-Image ID of annotations from file_names \n ---> {data_root}/{target}/{mode} \n---> Label is {class_}")
            tr_unique_num, val_unique_num = extracting_eigen_num(data_root,target=target)


            print(f"[Info msg] โจ Converting Coco Format Json to MMDetection Middle foramt Json \n ---> Working on {data_root}/{target}/{mode}_annotations.coco.json")
            print(f"---> Class is {class_} / Class Num is {int(dict_[class_])} \n")
            # f'{selff.crawled_root_dir}/weed_removal/valid/valid_annotations.coco.json
            coco2middle(open_json =f'{data_root}/{target}/{mode}_annotations.coco.json', 
                    unique_list = val_unique_num, # train or valid
                    class_of_this_object= int(dict_[class_]),
                    json_path_to_save=f'{data_root}/{target}/{mode}_label/',
                    mode=mode)


            '''
            << ๐ AI HUB 1๊ฐ์ ๋ฐ์ดํฐ์ ๋ํ json ํ์ผ ํ์ >>
            {
                "image": {
                    "date": "20210113",
                    "path": "S1-N06204M00001",
                    "filename": "S1-N06204M00893.jpg",
                    "copyrighter": "\ubbf8\ub514\uc5b4\uadf8\ub8f9\uc0ac\ub78c\uacfc\uc232(\ucee8)",
                    "H_DPI": 150,
                    "location": "06",
                    "V_DPI": 150,
                    "bit": "24",
                    "resolution": [
                        1920,
                        1080
                    ]
                },
                "annotations": [
                    {
                        "data ID": "S1",
                        "middle classification": "02",
                        "flags": {},
                        "box": [
                            1444,
                            857,
                            1619,
                            961
                        ],
                        "class": 4,
                        "large classification": "01"
                    },
                    {
                        "data ID": "S1",
                        "middle classification": "02",
                        "flags": {},
                        "box": [
                            1483,
                            858,
                            1529,
                            919
                        ],
                        "class": 4,
                        "large classification": "01"
                    },
                ]
            }

            <<๐ Roboflow์์ ์์ฑํ COCO ํ์์์ ์๋กญ๊ฒ ์์ฑํ json ํ์ผ ํ์ >>

            {
                "image": {
                    "resolution": [
                        1980,
                        1080
                    ],
                    "filename": "valid_special_vehicle_0.jpg"
                },
                "annotations": [
                    {
                        "class": 16,
                        "box": [
                            107,
                            660,
                            1603,
                            281
                        ]
                    }
                ]
            }

            '''
            print(f"\n [Info msg] โจ All Procedure of crawled_preprocess_format_conversion  has been DONE \n")


        def move_to_original_dataset(self):
            
            def movefile(filelist, movelist):
                for idx, filepath in enumerate(tqdm(filelist)):
                    copyfile(filepath, movelist[idx])

            DATA_PATH = f'{self.crawled_root_dir}'
            CLASS_NAME = 'weed_removal' #road_sweeper, special_vehicle, step_car, fire_truck, weed_removal


            # ํฌ๋กค๋ง ์ ์ฒ๋ฆฌ ์ด๋ฏธ์ง์ ํ์ผ๋ช ์ถ์ถ
            def extract(data_path, class_name, mode):

                extracted_list = os.listdir(os.path.join(data_path,class_name,mode))
                extracted_list = sorted(extracted_list)
                return extracted_list

            print(f"[Info msg] โจ Extracting Train & Valid Data & Labels files and Saving into lists \n")
            tr_filelist = extract(DATA_PATH, CLASS_NAME, 'train')
            tr_labellist = extract(DATA_PATH,CLASS_NAME,'train_label')
            val_filelist = extract(DATA_PATH,CLASS_NAME,'valid')
            val_labellist = extract(DATA_PATH,CLASS_NAME,'valid_label')


            TARGET_PATH = f"{self.original_root_dir}/"


            def move_to_original(target_path, mode, original_img_list, original_label_list, img_file_list, label_file_list):

                print(f"[Image] {mode}")
                movefile([os.path.join(filename) for filename in original_img_list], [os.path.join(target_path, f"{mode}_data", filename) for filename in img_file_list])

                print(f"[Label] {mode}")
                movefile([os.path.join(filename) for filename in original_label_list], [os.path.join(target_path, f'{mode}_label', filename) for filename in label_file_list])

            weed_tr_imgs = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/{CLASS_NAME}/train/*'))))
            weed_tr_labels = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/{CLASS_NAME}/train_label/*'))))
            weed_val_imgs = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/{CLASS_NAME}/valid/*'))))
            weed_val_labels = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/{CLASS_NAME}/valid_label/*'))))


            print(f"[Info msg] โจ Moving ' Class : <{CLASS_NAME}>   Train & Valid Data and Labels ' to ' Original Dataset '\n --> Original Dataset Path : {TARGET_PATH}/Train or Valid_data or label/ \n")

            move_to_original(
                            target_path = TARGET_PATH,
                            mode = 'train',
                            original_img_list = weed_tr_imgs,
                            original_label_list = weed_tr_labels,
                            img_file_list= tr_filelist,
                            label_file_list= tr_labellist)

            move_to_original(
                            target_path = TARGET_PATH,
                            mode = 'valid',
                            original_img_list = weed_val_imgs,
                            original_label_list = weed_val_labels,
                            img_file_list= val_filelist,
                            label_file_list= val_labellist)


            print(f"[Info msg] โจ Check Quantity of Original Dataset after working \n")
            # Original Dataset ์๋ ์ต์ข ํ์ธ
            tr_imgs = sorted(list(glob(os.path.join(f'{self.original_root_dir}/train_data/*'))))
            tr_labels = sorted(list(glob(os.path.join(f'{self.original_root_dir}/train_label/*'))))

            val_imgs = sorted(list(glob(os.path.join(f'{self.original_root_dir}/valid_data/*'))))
            val_labels = sorted(list(glob(os.path.join(f'{self.original_root_dir}/valid_label/*'))))

            test_imgs = sorted(list(glob(os.path.join(f'{self.original_root_dir}/test_data/*'))))
            test_labels = sorted(list(glob(os.path.join(f'{self.original_root_dir}/test_label/*'))))

            print(f"Train Data / Label : {len(tr_imgs)} / {len(tr_labels)}")
            print(f"Valid Data / Label : {len(val_imgs)} / {len(val_labels)}")
            print(f"Test Data / Label : {len(test_imgs)} / {len(test_labels)}")

            print(f"\n [Info msg] โจ All Procedure of move_to_original_dataset  has been DONE \n")
    

    # ์คํ
    conversion = Conversion()
    conversion.class_conversion()
    conversion.crawled_preprocess_format_conversion()
    conversion.move_to_original_dataset()


if __name__ == '__main__':
    main()
