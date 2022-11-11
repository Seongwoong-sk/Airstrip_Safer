# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/custom.py 여기에 있는 CustomDataset 클래스

'''
🔥 CustomDataset --> Middle Format Transformation 🔥 

👉 하나의 이미지에 하나의 annotation으로 매칭되고 있는데, custom에 맞게 변경.

👉 mmdetection의 중립 annotation 포맷 변환. 해당 포맷은 텍스트로 변환하지 않음. 바로 메모리 상의 list로 생성됨.

👉 filename, width, height, ann을 Key로 가지는 Dictionary를 이미지 개수대로 가지는 list 생성.
    ▶ filename : 이미지 파일명 (디렉토리는 포함하지 않음)
    ▶ width : 이미지 너비
    ▶ height : 이미지 높이
    ▶ ann : bounding box와 label에 대한 정보를 가지는 dictionary
           ▷ bboxes : 하나의 이미지에 있는 여러 Object 들의 numpy array. 4개의 좌표값(좌상단, 우하단)을 가지고, 해당 이미지에 n개의 Object들이 있을 경우 array의 shape는 (n, 4)
           ▷ labels: 하나의 이미지에 있는 여러 Object들의 numpy array. shape는 (n, )
           ▷ bboxes_ignore: 학습에 사용되지 않고 무시하는 bboxes. 무시하는 bboxes의 개수가 k개이면 shape는 (k, 4)
           ▷ labels_ignore: 학습에 사용되지 않고 무시하는 labels. 무시하는 bboxes의 개수가 k개이면 shape는 (k,)

👉 Example
   [
       {
           'filename' : 'a.jpg',
           'width' : 1280,
           'height' : 720,
           'ann' : {
               'bboxes' : <np.ndarray> (n, 4),
               'labels' : <np.ndarray> (n, ),
               'bboxes_ignore' : <np.ndarray> (k, 4),  # (optional field)
               'labels_ignore" : <np.ndarray> (k, 4)   # (optional field)
           }
       },
       ...
   ]

'''



import cv2
import mmcv
import json
import copy
import os.path as osp
from tqdm import tqdm
import numpy as np

from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.builder import DATASETS

@DATASETS.register_module(force=True)
class AirplaneDataset(CustomDataset):
    
   # 기존 AI_HUB의 CLASSES 에서 CLASSES 개수 변환됨 
    CLASSES = ['Aircraft','Rotocraft','Road surface facility','Obstacle (FOD)','Bird','Mammals','Worker',
               'Box','Pallet','Toyinka','Ramp bus','Step car','Fire truck','Road sweeper','Weeding vehicle',
               'Special vehicle','Forklift','Cargo loader','Tug Car'] 


    # annotation에 대한 모든 파일명을 가지고 있는 텍스트 파일을 __init__(self, ann_file)로 입력 받고, 
    # 이 self.ann_file이 load_annotations()의 인자로 입력
    def load_annotations(self, ann_file):

        print('##### self.data_root:', self.data_root, 'self.ann_file:', self.ann_file, 'self.img_prefix:', self.img_prefix)
        print('#### ann_file:', ann_file)

        label2cat = {i:k for i, k in enumerate(self.CLASSES)}
        image_list = mmcv.list_from_file(self.ann_file)  # ann_file을 다 받아서 리스트를 만듬
        mode_name = self.img_prefix.split('/')[-1].split('_')[0]
        
        # Middle Format 데이터를 담을 list 객체
        data_infos = []

        for image_id in tqdm(image_list, desc= 'Making Middle Format per Image'): 
            
            # self.img_prefix: 절대경로
            # 절대경로가 필요한 이유 : opencv imread를 통해서 이미지의 height, width 구함
            filename = f'{self.img_prefix}/{image_id}.jpg' 

            # 원본 이미지의 width, height 를 직접 로드하여 구함.
            image = cv2.imread(filename)
            height, width = image.shape[:2] #  height, width
            
            # 개별 image의 annotation 정보 저장용 Dict 생성. key값 filename 에는 image의 파일명만 들어감(디렉토리는 제외)
            # 이미지 하나는 하나의 data_info를 가지게 됨
            data_info = {'filename' : str(image_id) + '.jpg',
                         'width' : width, 'height' : height}


            # 개별 annotation이 있는 서브 디렉토리의 prefix 변환. 
            # annotation 정보는 label folder에서 가지고 있음
            label_prefix = self.img_prefix.replace(f'{mode_name}_data',f'{mode_name}_label')

            # json 파일 로드해서 bbox 저장
            bbox_classid = []
            bboxes = []

            file = open(osp.join(label_prefix, str(image_id)+'.json'))
            jsonFile = json.load(file)
            jsonObject = jsonFile.get('annotations')

            for j_list in jsonObject:
                bbox = j_list.get('box')[:4]
                cls_id = int(j_list.get('class')) # 1부터 클래스 아이디 불러옴

                # bbox 좌표를 저장
                bboxes.append(bbox)

                # 오브젝트의 클래스 id 저장
                # json 파일 내 클래스가 1부터 시작해서 0부터 시작하는 걸로 변경
                bbox_classid.append(cls_id-1)

            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []

            ## 위 이미지에서 key 'ann'을 만드는 작업
            # loop로 한번에 담기 위해서 생성
            for class_id, bbox in zip(bbox_classid, bboxes):

                if class_id in label2cat:
                    gt_bboxes.append(bbox) # 리스트의 리스트

                    gt_labels.append(class_id)  # label이 int로 이뤄져야되서 int인 클래스 아이디 그대로 불러옴. / 클래스별 int로 label array를 넘겨주면 mmdetection에서 알아서 클래스 별 아이디를 만들어서 할당

                else: # don't care (class에 포함되지 않는 것)은 여기에 집어넣기
                    gt_bboxes_ignore.append(bbox)
                    gt_labels_ignore.append(-1)
        
            # 개별 image별 annotation 정보를 가지는 Dict 생성. 해당 Dict의 value값은 모두 np.array임. 
            # 위의 것들을 한꺼번에 담는 annotation를 만듬 ->   위에서 작업한 것들을 한 middle format의 ann을 만드는 중
            data_anno = {
                'bboxes': np.array(gt_bboxes, dtype=np.float32).reshape(-1,4),
                'labels': np.array(gt_labels), # 1차원
                'bboxes_ignore' : np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1,4),
                'labels_ignore' : np.array(gt_labels_ignore, dtype=np.compat.long)
            }

            # image에 대한 메타 정보를 가지는 data_info Dict에 'ann' key값으로 data_anno를 value로 저장. 
            data_info.update(ann=data_anno)  # 위에서 만든 data_info dict에 ann이라는 키와 data_anno를 value로 추가함
            
            # 전체 annotation 파일들에 대한 정보를 가지는 data_infos에 data_info Dict를 추가
            data_infos.append(data_info)

        return data_infos # 리스트 객체

        
