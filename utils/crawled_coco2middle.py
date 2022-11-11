
# 👉 Roboflow에서 크롤링해서 얻은 이미지들로 데이터셋을 생성했었음.
# 👉 생성된 데이터셋의 Label 형식은 "COCO Format" 형식으로 돼있어서 MMDetection의 "Middle Format"으로 변환해서 Original Dataset으로 옮기는 파이썬 파일
# 👉 AI HUB의 원본 데이터셋의 Label 형식은 Pascal VOC 처럼 각 이미지마다 json 파일이 형성돼있음.

# 1️⃣ Class 4 (Road Surface disorder)와 Class 5 (Obstacle)은 FOD로 병합할 수 있어서 클래스 FOD로 병합
# --> 전체 데이터의 json 파일에 접근해서 클래스 id 변경

# 2️⃣ 크롤링한 데이터 전처리 후 추가
# --> 원본 json 파일 내에 기록되어 있는 image id를 이용하여 img & json 파일명 변경
# --> 고유번호 이용해서 coco json 파일 한 개에서 ann img마다 대응되는 json 파일 생성
# --> 원본 json 파일 내에 훈련에 불필요한 데이터들은 새롭게 만든 json 파일 내에 추가 시키지 않음

'''
< Structure Example >
├── airplane_custom (Root_Dir)
│   ├── Addition (Crawled Dir)
│         ├── fire_truck (classes)
│               ├── train
│               ├── train_label
│               ├── valid
│               ├── valid_label
│               ├── (test)
│               ├── (test_label)
│               ├── train_annotations_coco.json
│               ├── valid_annotations_coco.json
│               ├── (test_annotations_coco.json)
│         ├── special_vehicle
│               ├── ""
│         ├── step_car
│               ├── ""
│         ├── road_sweeper
│               ├── ""
│         ├── weed_removal
│               ├── ""
│
│   ├── 21000_Dataset (Original Dataset)
│         ├── train_data
│         ├── train_label
│         ├── valid_data
│         ├── valid_label
│         ├── test_data
│         ├── test_label
│         ├── train.txt (Annotation file)
│         ├── val.txt (Annotation file)
│         ├── test.txt (Annotation file)

'''

###################################### 👉 Libraries 👈 ################################################################

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


    ###################################### 👉 클래스 병합 👈 ################################################################

        def class_conversion(self):

            print(f"[Info msg] ✨ Class Combination Start 4 + 5 --> 4 \n")
            ## Json상 Class 4 (Road surface disorder) + Class 5 (Obstacle) 병합
            ## ---> Class 4는 FOD로 변경
            ## 최종 19개

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


            # class_num : class_name dict 생성
            global dict_ # coco2middle 함수에서 사용하기 때문에 전역 변수로 선언
            dict_ = {i+1:k for i,k in enumerate(CLASSES)}

            # 클래스가 1~20으로 이루어져있어서 5~20을 -1씩 하면 1~19로 됨.
            def class_change(label_file,change_start_idx, change_end_idx , mode=None):

                print(f'--- Start Changing class from {change_start_idx} to {change_end_idx}  ---')

                cnt=0
                change_range_ = [i for i in range(change_start_idx, change_end_idx+1)]
                change_cnt=0

                for i in tqdm(range(len(label_file)), desc= f'Changing Class in <{mode}> json files'): 

                    with open(label_file[i], 'r') as f:
                        jsonFile = json.load(f)

                        # annotations에 접근해서 class를 -1씩 만듦.
                        for ann_data in jsonFile['annotations']:
                            ann_data['class'] = int(ann_data['class'])
                            if ann_data['class'] in change_range_:
                                ann_data['class'] = int(ann_data['class'])-1 # -1
                                cnt+=1
                                
                    with open(label_file[i], 'w', encoding='utf-8') as file_:
                        json.dump(jsonFile, file_, indent='\t')
                        # print('json 파일 수정 완료')
                        change_cnt +=1
                
                print(f'{mode}_Class가 바뀐 횟수 : {cnt} \n {mode}_Json이 바뀐 횟수 : {change_cnt}')


            # 해당 클래스 id가 json 파일에 있는지 확인하는 함수
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
                
                print(f'{mode} Class ({class_to_check}) 개수 : {cnt}')

                return list_

            
            ### Data Checking ###
            print(f"[Info msg] ✨ Check Quantity of Original Dataset \n ---> Original Data Path : {self.original_root_dir}/train or valid or test_data or label/")
            tr_imgs = sorted(list(glob(os.path.join(f'{self.original_root_dir}/train_data/*'))))
            tr_labels = sorted(list(glob(os.path.join(f'{self.original_root_dir}/train_label/*'))))

            val_imgs = sorted(list(glob(os.path.join(f'{self.original_root_dir}/valid_data/*'))))
            val_labels = sorted(list(glob(os.path.join(f'{self.original_root_dir}/valid_label/*'))))

            test_imgs = sorted(list(glob(os.path.join(f'{self.original_root_dir}/test_data/*'))))
            test_labels = sorted(list(glob(os.path.join(f'{self.original_root_dir}/test_label/*'))))

            print(f"Quantity of Train Data : {len(tr_imgs)} / Quantity of Train Label : {len(tr_labels)}") 
            print(f"Quantity of Valid Data : {len(val_imgs)} / Quantity of Valid Label : {len(val_labels)}")
            print(f"Quantity of Test DAta : {len(test_imgs)} / Quantity of Test Label : {len(test_labels)} \n")



            ### class 5부터 20만 -1씩하면 1,2,3,4,5,6,.....,19로 완성 ###
            print(f"[Info msg] ✨ Change Train class 5 ~ 20 into 4 ~ 19 in json files \n")
            class_change(tr_labels,5,20,'Train')
            print(f"[Info msg] ✨ Change Valid class 5 ~ 20 into 4 ~ 19 in json files\n")
            class_change(val_labels,5,20,'Valid')
            print(f"[Info msg] ✨ Change Test class 5 ~ 20 into 4 ~ 19 in json files\n")
            class_change(test_labels,5,20,'Test')

            print(f"[Info msg] ✨  Checking whether class 20 exists or not in json files(20 Must not exist) \n")
            tr_checking_20 = class_checking(tr_labels,20,'Train')
            val_checking_20 = class_checking(val_labels,20,'Valid')
            test_checking_20 = class_checking(test_labels,20,'Test')
            


    ############################# 👉 크롤링한 데이터 전처리 후 오리지널 데이터셋에 추가 👈 ###################################

        def crawled_preprocess_format_conversion(self):

            print(f"[Info msg] ✨ Start Adding Crawled Data to the Original Dataset \n")


            ### Loading & Checking Quantity of Crawled Data into lists ###
            print(f"[Info msg] ✨ Loading & Checking Quantity of Crawled Data into lists \n")
            print(f"[Info msg] ✨ Train Data")
            t_fire = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/fire_truck/train/*'))))
            t_special = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/special_vehicle/train/*'))))
            t_step = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/step_car/train/*'))))
            t_road = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/road_sweeper/train/*'))))
            t_weed = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/weed_removal/train/*'))))

            print('Train : ', len(t_fire), len(t_special), len(t_step), len(t_road), len(t_weed))
            print(f"Train Sum : {sum([len(t_fire), len(t_special), len(t_step), len(t_road), len(t_weed)])}")

            print(f"\n [Info msg] ✨ Valid Data")
            v_fire = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/fire_truck/valid/*'))))
            v_special = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/special_vehicle/valid/*'))))
            v_step = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/step_car/valid/*'))))
            v_road = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/road_sweeper/valid/*'))))
            v_weed = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/weed_removal/valid/*'))))

            print('Valid : ', len(v_fire), len(v_special), len(v_step), len(v_road), len(v_weed))
            print(f"Valid Sum : {sum([len(v_fire), len(v_special), len(v_step), len(v_road), len(v_weed)])}")



            ### Roboflow Coco 형식인 json 파일 내에 기록되어 있는 image id를 이용하여 img & json 파일명 변경 ###
            '''
            << Roboflow에서 추출한 COCO 형식의 json 파일 >> 

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
            {'id': 0, 👈 이걸 활용 👈 
            'license': 1,
            'file_name': 'airport-step-car_173_jpg.rf.3d8188ea5df2295147ff0e09e8adbc30.jpg',
            'height': 480,
            'width': 640,
            'date_captured': '2022-09-05T12:47:53+00:00'},
            {'id': 1, 👈 이걸 활용 👈
            'license': 1,
            'file_name': 'airport-step-car_6_jpg.rf.566299ef20aa0ec399bf14d3c087bcd0.jpg',
            'height': 267,
            'width': 400,
            'date_captured': '2022-09-05T12:47:53+00:00'},
            ]

            'annotations': [
            {'id': 0,
            'image_id': 0, 👈 이걸 활용 👈
            'category_id': 1,
            'bbox': [149, 21, 354.59, 446.36],
            'area': 158274.7924,
            'segmentation': [],
            'iscrowd': 0},
            {'id': 1,
            'image_id': 1, 👈 이걸 활용 👈
            'category_id': 1,
            'bbox': [57, 16, 289.24, 214.78],
            'area': 62122.9672,
            'segmentation': [],
            'iscrowd': 0},
            ]
            }
            '''

            # json 내 image id에 접근해서 id가 포함된 파일명으로 변경 후 크롤링 폴더 경로에 저장하는 함수
            def change_filename_save(json_file, data_root, target_root, mode):

                cnt = 0
                change = 0
                finish = 0

                with open(json_file,'r') as f:

                    jsonFile = json.load(f)

                    # images 키에 접근
                    for img_data in tqdm(jsonFile['images']):

                        # images 키에 있는 각 이미지의 id를 추출
                        img_id = int(img_data['id'])
                        
                        file_name = img_data['file_name']
                        file_name = file_name[:file_name.rfind('.')]

                        # annotations 키에 있는 이미지 id를 추출 (images 키에 있는 id랑 동일)
                        for anno_data in jsonFile['annotations']:

                            if img_id == anno_data['image_id']:
                                print(f'\n\n ★★ image_id = {img_id} 이미지 ★★')
                                print(f"Original filename : {img_data['file_name']} //// Original img_id : {img_id} ")

                                print(' json 내 파일명 변경!!!!!  ')

                                # 실제 바꿀 이름명. 이름 변경
                                img_data['file_name'] = f'{mode}_{target_root}_{str(img_id)}.jpg'
                                print(f'--> json 내 파일명 변경 완료!!  :::', img_data['file_name'], '\n')
                                cnt+=1

                                print('--> Json 내 변경된 파일명으로 실제 이미지 이름 변경!!')
                                

                                if not os.path.exists(os.path.join(data_root,target_root,mode,img_data['file_name'])):
                                    # 이미지 변경해서 정해진 경로에 저장
                                    os.rename(os.path.join(data_root, target_root, mode, file_name+'.jpg'),
                                            os.path.join(data_root, target_root, mode, img_data['file_name']))
                                    
                                    print(f" 변경 완료!!!\n--> 변경된 이미지 경로 : {os.path.join(data_root, target_root, mode, img_data['file_name'])}")
                                    change+=1


                # 기존 json 파일 덮어쓰기
                # annotation이 있는 이미지만 추려서 파일명 변경했음.
                with open(json_file, 'w', encoding='utf-8') as file_:
                    json.dump(jsonFile, file_, indent="\t")
                    print('json 파일 최종 수정 완료!!')
                    finish +=1
                
                print(f"json 내 파일명 변경 변경된 횟수 : {cnt} // 이미지 ID으로 파일명 변경된 횟수 : {change} // 최종 수정된 파일 갯수 : {finish}")


            DATA_ROOT = f'{self.crawled_root_dir}'
            TARGET_ROOT = 'fire_truck' # road_sweeper, special_vehicle, step_car, fire_truck, weed_removal
            MODE = 'valid' # train or valid

            print(f"[Info msg] ✨ Start Changing file_names in jsons and save to crawled folder \n")
            change_filename_save(json_file=f'{DATA_ROOT}/{TARGET_ROOT}/{MODE}_annotations.coco.json', 
                        data_root = DATA_ROOT,
                        target_root=TARGET_ROOT, 
                        mode=MODE)


            # Image ID (고유번호) 이용해서 coco json 파일 한 개에서 ann img마다 대응되는 json 파일 생성

            tr_img_lists = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/fire_truck/train/*'))))
            val_img_lists = sorted(list(glob(os.path.join(f'{self.crawled_root_dir}/fire_truck/valid/*')))) 

            tr_truc_elems = os.listdir(os.path.join(f'{self.crawled_root_dir}/weed_removal/train'))
            val_truc_elems = os.listdir(os.path.join(f'{self.crawled_root_dir}/weed_removal/valid'))

            print(val_truc_elems[1]) # valid_weed_removal_1.jpg
            print(val_truc_elems[1].split('_')[-1].split('.')[0]) # 1



            # 이미지명에서 고유 id 꺼내서 리스트에 저장하는 함수
            def extracting_eigen_num(data_root,target,tr_mode='train',val_mode='valid'):
                '''
                < args >
                data_root : 크롤링한 데이터들을 폴더별로 구분해놓은 경로
                target : 크롤링한 이미지들의 라벨 이름 (ex : step_car, weed_removal)
                tr_mode : Train Data Folder Accessing
                val_mode : Valid Data Folder Accessing
                '''

                tr_truc_elems = os.listdir(os.path.join(f'{data_root}/{target}/{tr_mode}'))
                val_truc_elems = os.listdir(os.path.join(f'{data_root}/{target}/{val_mode}'))

                # 이미지명에서 고유 id 꺼내기

                tr_unique_num = []
                val_unique_num = []

                for i in range(len(tr_truc_elems)):

                    num = int(tr_truc_elems[i].split('_')[-1].split('.')[0]) # ex : valid_weed_removal_1.jpg --> 1
                    tr_unique_num.append(num)
                
                for k in range(len(val_truc_elems)):

                    val_num = int(val_truc_elems[k].split('_')[-1].split('.')[0])
                    val_unique_num.append(val_num)

                print(f"✨[Info msg] Unique ID of Train / Valid Images : {len(tr_unique_num)} / {len(val_unique_num)}")

                return tr_unique_num, val_unique_num


            # 이미지와 객체에 대한 정보가 하나의 json 파일에 다 들어있는 COCO 형식 json에 접근하여 
            # Image ID를 이용해 한 이미지에 대응하는 하나의 json 파일들을 생성하는 함수
            def coco2middle(open_json, unique_list, class_of_this_object,json_path_to_save,mode):

                '''
                <args> 
                open_json : 원본 coco json 파일
                unique_list : annotation이 있는 이미지의 파일명에 있는 num들이 모여져있는 list
                class_of_this_object : 지정할 클래스
                json_path_to_save : json files을 저장할 경로
                mode : train or valid
                '''

                with open(open_json,'r') as f:
                    ann_data = json.load(f)
                    cnt=0

                    for img_data in ann_data['images']:


                        img_id = int(img_data['id'])
                        file_num = int(img_data['file_name'].split('_')[-1].split('.')[0]) # 파일명에 잇는 이미지 번호
                        
                        for anno_data in tqdm(ann_data['annotations']):

                            if (img_id == anno_data['image_id']) and (file_num in unique_list):

                                print(f'{mode} file_num : {file_num} 차례입니다 // img_id : {img_id}')
                                cnt+=1

                            # 변경된 파일명에 대응하는 json 파일을 만들고 그 파일에 맞는 ann 정보 입력해서 json 새로 생성

                                file_name = img_data['file_name'].split('.')[0]
                                if not os.path.exists(json_path_to_save): os.mkdir(json_path_to_save)
                                label_path = json_path_to_save


                                with open(os.path.join(label_path,file_name+'.json'),'w') as f_out:


                                # json 파일 내용 넣을 곳

                                    data_dict = {} # base format

                                    data_dict['image'] = {}  # base format은 이미지 1장  :: dict
                                    data_dict['annotations'] = [] # 이미지 1장에 있는 여러 개의 객체 :: list
                                    data_dict['image']['resolution']= [0 for _ in range(2)]  # resolution 담을 2 크기의 리스트 생성

                                    data_dict['image']['filename'] = img_data['file_name']
                                    data_dict['image']['resolution'][0]= img_data['width']
                                    data_dict['image']['resolution'][1] = img_data['height']

                                    base_format = {} # annotations list안에 넣을 여러 개의 dic
                                    base_format['class'] = class_of_this_object

                                    # coco 형태 [x1,y1,bbox_w,bbox_h] 를 프로젝트 bbox 형식인 [x1,y1,x2,y2]로 변경
                                    # BBox 형태를 기존 COCO형태 [x1,y1,bbox_w,bbox_h]로 넣어서 변환한 json 접근해서 bbox 값 수정
                                    base_format['box'] = [0 for _ in range(4)] # # bbox 담을 4 크기의 리스트 생성
                                    base_format['box'][0] = anno_data['bbox'][0] # x1
                                    base_format['box'][1] = anno_data['bbox'][1] # y1
                                    base_format['box'][2] = anno_data['bbox'][0] + anno_data['bbox'][2]# x1 + bbox_w
                                    base_format['box'][3] = anno_data['bbox'][1] + anno_data['bbox'][3]# x1 + bbox_w

                                    data_dict['annotations'].append(base_format)
                                                
                                    json.dump(data_dict , f_out, indent='\t') # img명을 가진 json 파일 각각 생성
                    
                    print(f"{cnt}개 성공!!")


            data_root = f'{self.crawled_root_dir}'
            target = 'weed_removal' #road_sweeper, special_vehicle, step_car, fire_truck, weed_removal
            mode = 'valid' # train
            class_ = 'Weeding vehicle' # 'Road sweeper', 'Special vehicle', 'Step car', 'Weeding vehicle', 'Fire truck'


            print(f"[Info msg] ✨ Extracting each eigen-Image ID of annotations from file_names \n ---> {data_root}/{target}/{mode} \n---> Label is {class_}")
            tr_unique_num, val_unique_num = extracting_eigen_num(data_root,target=target)


            print(f"[Info msg] ✨ Converting Coco Format Json to MMDetection Middle foramt Json \n ---> Working on {data_root}/{target}/{mode}_annotations.coco.json")
            print(f"---> Class is {class_} / Class Num is {int(dict_[class_])} \n")
            # f'{selff.crawled_root_dir}/weed_removal/valid/valid_annotations.coco.json
            coco2middle(open_json =f'{data_root}/{target}/{mode}_annotations.coco.json', 
                    unique_list = val_unique_num, # train or valid
                    class_of_this_object= int(dict_[class_]),
                    json_path_to_save=f'{data_root}/{target}/{mode}_label/',
                    mode=mode)


            '''
            << 👉 AI HUB 1개의 데이터에 대한 json 파일 형식 >>
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

            <<👉 Roboflow에서 생성한 COCO 형식에서 새롭게 생성한 json 파일 형식 >>

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
            print(f"\n [Info msg] ✨ All Procedure of crawled_preprocess_format_conversion  has been DONE \n")


        def move_to_original_dataset(self):
            
            def movefile(filelist, movelist):
                for idx, filepath in enumerate(tqdm(filelist)):
                    copyfile(filepath, movelist[idx])

            DATA_PATH = f'{self.crawled_root_dir}'
            CLASS_NAME = 'weed_removal' #road_sweeper, special_vehicle, step_car, fire_truck, weed_removal


            # 크롤링 전처리 이미지의 파일명 추출
            def extract(data_path, class_name, mode):

                extracted_list = os.listdir(os.path.join(data_path,class_name,mode))
                extracted_list = sorted(extracted_list)
                return extracted_list

            print(f"[Info msg] ✨ Extracting Train & Valid Data & Labels files and Saving into lists \n")
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


            print(f"[Info msg] ✨ Moving ' Class : <{CLASS_NAME}>   Train & Valid Data and Labels ' to ' Original Dataset '\n --> Original Dataset Path : {TARGET_PATH}/Train or Valid_data or label/ \n")

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


            print(f"[Info msg] ✨ Check Quantity of Original Dataset after working \n")
            # Original Dataset 수량 최종 확인
            tr_imgs = sorted(list(glob(os.path.join(f'{self.original_root_dir}/train_data/*'))))
            tr_labels = sorted(list(glob(os.path.join(f'{self.original_root_dir}/train_label/*'))))

            val_imgs = sorted(list(glob(os.path.join(f'{self.original_root_dir}/valid_data/*'))))
            val_labels = sorted(list(glob(os.path.join(f'{self.original_root_dir}/valid_label/*'))))

            test_imgs = sorted(list(glob(os.path.join(f'{self.original_root_dir}/test_data/*'))))
            test_labels = sorted(list(glob(os.path.join(f'{self.original_root_dir}/test_label/*'))))

            print(f"Train Data / Label : {len(tr_imgs)} / {len(tr_labels)}")
            print(f"Valid Data / Label : {len(val_imgs)} / {len(val_labels)}")
            print(f"Test Data / Label : {len(test_imgs)} / {len(test_labels)}")

            print(f"\n [Info msg] ✨ All Procedure of move_to_original_dataset  has been DONE \n")
    

    # 실행
    conversion = Conversion()
    conversion.class_conversion()
    conversion.crawled_preprocess_format_conversion()
    conversion.move_to_original_dataset()


if __name__ == '__main__':
    main()
