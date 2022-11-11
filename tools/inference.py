import torch
import os
import sys
import cv2
import time
    

################################################################################################################
import mmcv
import argparse
from mmcv import Config
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--config', help='Choose to use a config and initialize the detector')
    parser.add_argument('--checkpoint', help='Setup a checkpoint file to load')
    parser.add_argument('--device', default='cuda:0', help='Set the device to be used for evaluation')
    parser.add_argument('--img', default=None, help='Set an img path to infer')
    parser.add_argument('--video', default=None, help='Set an video path to infer')
    parser.add_argument('--threshold', type=float,default=0.5, help='Set a threshold score to infer')
    parser.add_argument('--out_img', type=str, help='Output img file')
    parser.add_argument('--out_video', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
    parser.add_argument('--wait-time',type=float,default=1,help='The interval of show (s), 0 is block')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():

    args = parse_args()
    if args.video is not None:
        assert args.out_video or args.show, \
            ('Please specify at least one operation (save/show the '
            'video) with the argument "--out_video" or "--show"')
    assert args.img or args.video, \
        ('Please specify at least one element Img or Video')

    # Load the config
    cfg = mmcv.Config.fromfile(args.config)

    # Initialize the detector
    model = init_detector(cfg, args.checkpoint, device=args.device)


    # We need to set the model's cfg for inference
    model.cfg = cfg

    # Image
    if args.img is not None and args.video is None:
        # Use the detector to do inference
        img = cv2.imread(args.img)
        result = inference_detector(model, img)

        # plot the result
        if args.out_img:
            show_result_pyplot(model, img, result, score_thr=args.threshold, out_file=args.out_img)
        show_result_pyplot(model, img, result, score_thr=args.threshold)
        

    # Video
    elif args.img is None and args.video is not None:
        video_reader = mmcv.VideoReader(args.video)
        video_writer = None
        if args.out_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                args.out_video, fourcc, video_reader.fps,
                (video_reader.width, video_reader.height))
        btime = time.time()

        for frame in mmcv.track_iter_progress(video_reader):
            stime = time.time()
            result = inference_detector(model, frame)
            frame = model.show_result(frame, result, score_thr=args.threshold)
            if args.show:
                cv2.namedWindow('video', 0)
                mmcv.imshow(frame, 'video', args.wait_time)
            if args.out_video:
                video_writer.write(frame)

        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
