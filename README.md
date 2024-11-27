# 자율주행 차량의 안전 운행을 위한 CCTV 연계 보행자 돌발상황 감지
## Algorithm
![image](https://github.com/user-attachments/assets/a4bdd8e0-4132-4bb7-9365-3d3a1d83441e)  
> 1. Image Segmentation을 통해 Object Detection 데이터셋의 도로 영역을 분할
>     * Segmentation 모델은 FAR의 FO & FA 분기를 2D task에 맞게 변환하여 ResNet에 추가  
>     * 위의 ResNet을 backbone으로 사용한 DeepLabv3+ 모델을 최종적으로 사용 
> 2. 기존의 데이터셋에 도로 영역을 분할한 이미지를 추가하여 학습을 진행
>     * YOLO v8-m 모델을 사용하여 학습을 진행
---
## 예시 예측 이미지 
![image](https://github.com/user-attachments/assets/3e57bb11-67e0-45a8-b52f-9744e416430e)
