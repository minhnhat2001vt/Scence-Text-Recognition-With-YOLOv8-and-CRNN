# Scence Text Recognition With YOLOv8 and CRNN


Scene Text Recognition (STR) is a problem that applies image processing and character recognition algorithms to identify text appearing in images. A Scene Text Recognition program usually includes two main stages: Text Detection (Detector) and Text Recognition (Recognizer).

The goal of this project is to build a STR model that can takes an image containing text as input, then returns Coordinate location and text in the image as outputs.
## 1. Dataset
- The dataset using in this project is ICDAR 2003, which is a dataset for scene text recognition.
- Contains 507 natural scene images, (including 258 training images and 249 test images) in total.
- Dataset download at: https://drive.google.com/file/d/1x9e2FNDlKc_lBkJvHvWSKKfCSSqNsQfM/view
## 2. Data Preparation
## 3. Text Detector with YOLOv8
- Model summary: YOLOv8 is an advanced object detection model by Ultralytics that uses anchor-free detection
- Training:
## 4. Text Recognizer with CRNN
- Model summary:
- Training:
## 5. Evaluation
- YOLOv8:
- Plain CRNN:
- CRNN + Skip Connection:
- CRNN + Resnet

## 6. References
- 7th International Conference on Document Analysis and Recognition (ICDAR 2003), 2-Volume Set, 3-6 August 2003, Edinburgh, Scotland, UK. IEEE Computer Society 2003, ISBN 0-7695-1960-1
- 

 
