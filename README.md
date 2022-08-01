# Crop-disease-diagnosis

## 파일 경로

```bash
Crop-disease-diagnosis
│   README.md
│   utils.py
│
└───history
│   │   [Baseline]_ResNet50_+_LSTM.ipynb
│   │   add autoaugment_focus image.ipynb
│   │  add scheduler early stopping.ipynb
│   │	...
│
└───src
│   │	preprocessing.ipynb
│   │	train.ipynb
│   └──	inference.ipynb
│
└───input
    └──	train.csv
```


## 대회 설명
![image](https://user-images.githubusercontent.com/80466735/182122631-7b69a3ba-4c5a-4ad2-a1b6-abe50a78c356.png)


_Image.1 특정 잎 부분을 box로 표현했다. src/preprocessing.ipynb에서 데이터 분석한 결과를 볼 수 있다._


농업 환경 변화에 따른 작물 병해 진단 AI모델을 개발하는 대회다. 작물 환경 데이터, 병해 이미지를 이용해 작물의 종류, 병해의 종류, 진행정도를 진단하면 된다.

[대회 경로](https://dacon.io/competitions/official/235870/overview/description)


## 모델 요약
- Data Augmentation
  cutMix와 autoaugment를 이용했다. autoaugment가 큰 활약을 보이지는 않았다. 하지만 cutmix는 확실히 정확도 향상을 보였다.
- Model
  대회는 동작 속도를 보기 때문에 시간이 빠르고 무겁지 않은 모델들을 이용했다. 
  * Vision 모델
   densnet201과 resnet50이 가장 큰 활약을 했다. 하지만 efficientNet-b0나 Mobilenetv3-large는 큰 활약을 보였다. 
   
  * Time Series 모델
   LSTM만을 사용했다. LSTM의 layer를 3층 layer로 바꾸는 시도를 했다. 동작 속도의 큰 영향을 안끼쳤지만 acc 향상에 큰 영향을 못줬다.

## 사용법

- 학습시
모든 파일을 clone후 적절한 src/train.ipynb 파일에 있는 경로를 바꾸면 된다.(데이터셋은 위에 경로에서 다운)

- 모델 평가시
src/inference.ipynb 파일에서 경로 변경 후 사용한다. 깃허브에 모델을 저장 할수가 없었다. 만약 필요하다면 cjfghk5697@gmail.com으로 보내주겠습니다. 

## 느낀점
 대략 상위 10퍼 정도 결과를 얻을 수 있었다. 시계열 데이터와 이미지 데이터를 활용한 대회에서 꼭 시계열 데이터가 필요할 거 같지는 않다. 이번에 대회하면서 resampling, focus image, TTA, image 사이즈를 키워보는 등 여러시도를 해봤다. 하지만 전부 큰 효과를 보진 않았다.
 이외에도 preprocessing 하는 방법을 좀 깊게 공부하는 좋은 기회가 되었다. 

## 결론

![image](https://user-images.githubusercontent.com/80466735/182125400-dce3c088-d93e-4f1a-8a92-1a1c9c1a6e72.png)

_Image.2 결과적으로 private 91% 도달했다. 대략 상위 10% 정도의 결과다._
