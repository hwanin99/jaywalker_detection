#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import glob
import os
import time
from PIL import Image
from ultralytics import YOLO

def run_inference(model_path,test_dir,result_dir):
    model_path = args.model_path
    test_dir = args.test_dir
    result_dir = args.result_dir
    
    # 모델 정의
    model = YOLO(model_path)
    
    # 모델 예측 결과 저장
    results = model.predict(source=test_dir, project=result_dir, show_conf=False, save_txt=True, save=True)

    time.sleep(2)

    images = sorted(glob.glob(f'./{result_dir}/predict/*.jpg'),key=lambda x: os.path.splitext(x)[0])
    labels = sorted(glob.glob(f'./{result_dir}/predict/labels/*.txt'),key=lambda x: os.path.splitext(x)[0])

    # 이미지와 라벨을 순서대로 매칭하여 변환
    for image_path, label_path in zip(images, labels):
        image_path = image_path.replace('\\','/')
        label_path = label_path.replace('\\','/')
        # 이미지 크기 가져오기
        w, h = Image.open(image_path).size

        # 변환된 좌표를 저장할 리스트
        voc_labels = []

        # 라벨 파일 읽기
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # YOLO 형식 데이터 파싱
                label, x_center, y_center, box_width, box_height = line.split()

                # VOC 형식 좌표 계산
                x_min = round(w * max(float(x_center) - float(box_width) / 2, 0),5)
                x_max = round(w * min(float(x_center) + float(box_width) / 2, 1),5)
                y_min = round(h * max(float(y_center) - float(box_height) / 2, 0),5)
                y_max = round(h * min(float(y_center) + float(box_height) / 2, 1),5)

                # 변환된 좌표를 리스트에 추가
                voc_labels.append(f"{label} {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f}")

        with open(label_path, 'w') as f:
            f.write("\n".join(voc_labels))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='교통CCTV 돌발상황 분석 프로그램')

    parser.add_argument('--model_path', type=str, required=True, help='모델 파일 경로를 입력하세요.')
    parser.add_argument('--test_dir', type=str, required=True, help='분석할 이미지 디렉토리 경로를 입력하세요.')
    parser.add_argument('--result_dir', type=str, required=True, help='분석한 결과가 저장될 디렉토리 이름을 입력하세요.')    

    args = parser.parse_args()
    
    run_inference(args.model_path, args.test_dir,args.result_dir)

