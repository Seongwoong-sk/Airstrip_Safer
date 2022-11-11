
################################ ✨ Libraries ✨ ##############################
import os
import sys
import cv2
import time
import mmcv
import mmdet
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import moviepy.editor as moviepy
from PIL import Image, ImageEnhance
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector


################################ ✨ Methods Definition ✨ ##############################

###### 🛠️ image를 입력으로 받아서 Inference하는 함수 🛠️ ######
################################################################
def get_detected_img(model, img_array,  score_threshold=0.5):

    '''
    < args >
    model : loaded model
    img_array : image array(numpy) to inference
    '''
    CLASSES = ['Aircraft','Rotocraft','Road surface facility','Obstacle (FOD)','Bird','Mammals','Worker',
               'Box','Pallet','Toyinka','Ramp bus','Step car','Fire truck','Road sweeper','Weeding vehicle',
               'Special vehicle','Forklift','Cargo loader','Tug Car']

    #  model과 원본 이미지 array, filtering할 기준 class confidence score를 인자로 가지는 inference 시각화용 함수 생성 #
    labels_to_names_seq = {i:k for i, k in enumerate(CLASSES)}

    # Detect한 objects 담을 리스트트
    objects_list = []

    # Confidence Score 담을 리스트
    confs_list = []
    
    # 인자로 들어온 image_array를 복사. 
    draw_img = img_array.copy()
    bbox_color=(255, 0, 153)
    text_color=(251, 0, 255) 

    # model과 image array를 입력 인자로 inference detection 수행하고 결과를 results로 받음. 
    # results는 19개(CLASS 개수)의 2차원 array(shape=(오브젝트갯수, 5))를 가지는 list. 
    results = inference_detector(model, img_array)

    # 19개의 array원소를 가지는 results 리스트를 loop를 돌면서 개별 2차원 array들을 추출하고 이를 기반으로 이미지 시각화 
    # results 리스트의 위치 index가 바로 Custom Dataset 매핑된 Class id. 여기서는 result_ind가 class id
    # 개별 2차원 array에 오브젝트별 좌표와 class confidence score 값을 가짐. 
    for result_ind, result in enumerate(results):  # -> 여기서 result는 2차원
    
        # 개별 2차원 array의 row size가 0 이면 해당 Class id로 값 (detect된 obj 개수)이 없으므로 다음 loop로 진행. 
        if len(result) == 0:
            continue
        
        # 2차원 array에서 5번째 컬럼에 해당하는 값이 score threshold이며 이 값이 함수 인자로 들어온 score_threshold 보다 낮은 경우는 제외. 
        result_filtered = result[np.where(result[:, 4] > score_threshold)]
        
        # 해당 클래스 별로 Detect된 여러 개의 오브젝트 정보가 2차원 array에 담겨 있으며, 이 2차원 array를 row수(detect된 obj 개수)만큼 iteration해서 개별 오브젝트의 좌표값 추출. 
        for i in range(len(result_filtered)):

            # 좌상단, 우하단 좌표 추출. 
            left = int(result_filtered[i, 0])
            top = int(result_filtered[i, 1])
            right = int(result_filtered[i, 2])
            bottom = int(result_filtered[i, 3])
            caption = "{}: {}%".format(labels_to_names_seq[result_ind], int(100*result_filtered[i, 4])) # 4 = confidence score

            # Detect된 각 object와 그것의 confidence score를 Tuple에 담아서 리스트에 저장
            objects_list.append(labels_to_names_seq[result_ind].upper())
            confs_list.append((int(100*result_filtered[i, 4])))

            cv2.rectangle(draw_img, (left, top), (right, bottom), color=bbox_color, thickness=2)
            cv2.putText(draw_img, caption, (int(left), int(top - 7)), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    return draw_img, objects_list, confs_list


###### 🛠️ Streamlit에서 image를 입력으로 받아서 Inference하는 함수 🛠️ ######
##############################################################################
def detection_image():
    st.title('📷 Image Object Detection 📷')
    st.markdown("")
    st.subheader('This page takes an image and return an image with bounding boxes created around the objects in the image.')
    
    # Model Weight 지정정
    cfg = mmcv.Config.fromfile('/content/drive/MyDrive/Air_PY/configs/FINAL.py')
    checkpoint_file = '/content/drive/MyDrive/Air_PY/checkpoints/final_weight.pth'
    model = init_detector(cfg, checkpoint_file, device='cuda:0')
    model.cfg = cfg
    CLASSES = ['Aircraft','Rotocraft','Road surface facility','Obstacle (FOD)','Bird','Mammals','Worker',
                'Box','Pallet','Toyinka','Ramp bus','Step car','Fire truck','Road sweeper','Weeding vehicle',
                'Special vehicle','Forklift','Cargo loader','Tug Car'] 
    model.CLASSES = CLASSES

    file = st.file_uploader('Upload Image', type = ['jpg','png','jpeg'])
    if file!= None:
        img1 = Image.open(file)
        
        img_arr = np.array(img1)
        st.image(img_arr, caption = "Uploaded Image", width=1920, use_column_width = True)
                
        my_bar = st.progress(0)

        confThreshold =st.slider('Confidence', 0, 100, 50)

        detected_img, objs, confs = get_detected_img(model, img_arr,  score_threshold=(confThreshold/100))
        df= pd.DataFrame(list(zip(objs,confs)),columns=['Object Name','Confidence (%)'])

        if st.checkbox("Show Object's list" ):
            st.write(df)
        if st.checkbox("Show Confidence bar chart" ):
            st.subheader('Bar chart for confidence levels')
            st.bar_chart(df["Confidence (%)"])
        
        st.image(detected_img, caption='Proccesed Image.',width=1920, use_column_width = True)
        my_bar.progress(100)


###### 🛠️ Streamlit에서 Video를 입력으로 받아서 Inference하는 함수 🛠️ ######
##############################################################################
def detection_video():
    Confidence_score = 0.5

    cfg = mmcv.Config.fromfile('/content/drive/MyDrive/Air_PY/configs/FINAL.py')
    checkpoint_file = '/content/drive/MyDrive/Air_PY/checkpoints/final_weight.pth'
    model = init_detector(cfg, checkpoint_file, device='cuda:0')
    model.cfg = cfg
    CLASSES = ['Aircraft','Rotocraft','Road surface facility','Obstacle (FOD)','Bird','Mammals','Worker',
                'Box','Pallet','Toyinka','Ramp bus','Step car','Fire truck','Road sweeper','Weeding vehicle',
                'Special vehicle','Forklift','Cargo loader','Tug Car'] 
    model.CLASSES = CLASSES

    st.title('🎥 Video Object Detection 🎥')
    st.markdown("")
    st.subheader('This page takes a video and return a video with bounding boxes created around the objects in the video.')
    
    uploaded_video = st.file_uploader("Upload Video", type = ['mp4','mpeg','mov'])
    if uploaded_video != None:

        vid = uploaded_video.name
        st.write(vid)
        
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read()) # save video to disk

        st_video = open(vid,'rb') # bytes 형식으로 출력력
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Uploaded Video")

        cap = cv2.VideoCapture(vid)
        _, image = cap.read()
        h, w = image.shape[:2]
        vid_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        vid_fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'XVID') # 
        vid_writer = cv2.VideoWriter("detected_video.mp4", fourcc, vid_fps, vid_size) 
        count = 0

        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.write('Total amount of Frames : ', frame_cnt)
        btime = time.time()

        my_bar = st.progress(0)
        while True:
            hasFrame, img_frame = cap.read()
            if not hasFrame:
                break
            
            img_frame,_,_ = get_detected_img(model, img_frame,  score_threshold=0.5)
        
            vid_writer.write(img_frame)
            count+=1
            progress =[round(x*0.1,1) for x in range(1, 11,1)] # 0.1,0.2,0.3,0.4...1.0
            for i in range(len(progress)):
                if count == int(frame_cnt*progress[i]):
                    my_bar.progress(int(progress[i]*100))

        ### end of while loop
        cap.release() # 재생 파일 종료
        vid_writer.release() # 저장 파일 종료료
        
        # st.write(f'⏰ 최종 detection 완료 수행 시간 : {int(round(time.time() - btime)/60)}m {int(round(time.time()-btime)%60)}s ⏰')
        # st.write(f'💥 FPS : {int(round(frame_cnt/int(round(time.time()-btime))),1)}')
        my_bar.progress(100)



################################ ✨ Main ✨ ##############################
def main():

    st.sidebar.title("Select Activity")
    choice  = st.sidebar.selectbox("MODE",("About","Object Detection(Image)","Object Detection(Video)"))


    if choice == 'About':
        title = st.markdown('# 🛫 Abnormal Object Detection on Airstrip 🛬')
        blank1 = st.markdown('')
        blank2 = st.markdown('')

        remark1 = st.markdown("##### This project was built using **MMDetection** and **Streamlit** to demonstrate Abnormal Object detection on Airstrip in both videos(pre-recorded) and images.")
        blank3 = st.markdown('')
        remark2 = st.markdown("##### This **Object Detection** project can detect 19 objects(i.e classes) in either a video or image. The full list of the classes can be found [here](https://github.com/KaranJagtiani/YOLO-Coco-Dataset-Custom-Classes-Extractor/blob/main/classes.txt)")
        blank4 = st.markdown('')
        remark3 = st.markdown("##### If you would like to know more details about MMDetection, you could visit [here](https://github.com/open-mmlab/mmdetection)")

    if choice == "Object Detection(Image)":
        detection_image()
            
    if choice == "Object Detection(Video)":
        detection_video()
        try:
            clip = moviepy.VideoFileClip('detected_video.mp4')
            clip.write_videofile("myvideo.mp4")
            st_video = open('myvideo.mp4','rb')
            video_bytes = st_video.read()
            st.video(video_bytes)
            st.write("Detected Video") 

        except OSError as e:
            st.write(e)


if __name__ == '__main__':
    main()
