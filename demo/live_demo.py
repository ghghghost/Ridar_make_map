# based on https://github.com/open-mmlab/mmdetection/blob/v2.28.0/demo/image_demo.py
# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet.apis import init_detector
import cv2

from libs.api.inference import inference_one_image
from libs.utils.visualizer import visualize_lanes


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default='result.png', help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold'
    )
    args = parser.parse_args()
    return args

def process_frame(model, frame):
    src, preds = inference_one_image(model, frame)
    # 결과 처리 (예: 화면에 출력)
    print(preds)
    return src, preds

def process_video_stream(model):
    cap = cv2.VideoCapture(0)  # 웹캠을 사용하려면 0으로 설정

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        src, preds = process_frame(model, frame)
        dst = visualize_lanes(src, preds)

        # 결과를 화면에 표시
        cv2.imshow('Inference', dst)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    process_video_stream(model)


if __name__ == '__main__':
    args = parse_args()
    main(args)
