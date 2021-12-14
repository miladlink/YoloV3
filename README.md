# Pytorch YoloV3 implementation from scratch

## Demo images

![view](https://user-images.githubusercontent.com/81680367/146061204-1e7057a6-6482-40b4-acb8-881708ec2473.jpg)
![home](https://user-images.githubusercontent.com/81680367/146061258-0c1fd0a4-30c9-4b12-8d2a-6fb29264552d.jpg)

## Description

* `YOLOv2` is fast algorithm for object detection but it's **less accurate** compare other object detection methods. Furthermore it has problem with **little objects** so it can not detect little object and multiple object in one place. So YOLO team release version 3 of their algorithem.

* `YOLOv3` is more accurate compare `YOLOv2` but slower than it, but stil fast and it can detect little objects (look Demo images)

* This repository is simple implementation of `YOLOv3 algorithm` for better understanding and use it for more **object detection** usage. This project based on Pytorch. The code of project is so easy and clear.

## Dataset

Pretrained weights in this implemetation are based on training yolo team on COCO trainval dataset

## Usage

**You can have your own object detection machine**

**Note**: The tests of this repo run with `cpu` mode so if you use `gpu` prediction become much faster

### Clone the repository

```bash
git clone https://github.com/miladlink/YoloV3.git

cd YoloV3
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Pretrained Weights

* Download used weights in this project from [here](https://pjreddie.com/media/files/yolov3.weights) or go to weights and click to yolov3.weights

**or**

```bash
chmod +x weights/get_weights.sh
weights/get_weights.sh
```

### Help

```bash
python detect.py -h
```

![image](https://user-images.githubusercontent.com/81680367/145953199-0addc1c0-d63d-4462-890d-10f6a9a8c8e4.png)

### Detection

```bash
python detect.py -w path/to/weights\
-ct <conf_thresh>\
-nt <nms_thresh>\
-p path/to/img\
-s
```

## Some Result image

you can see some examples in `yolov3_examples.ipynb`

![dog](https://user-images.githubusercontent.com/81680367/146063007-f60cfcda-e517-4255-a6c6-bb711006f387.jpg)

![horses](https://user-images.githubusercontent.com/81680367/146063037-ea57e777-5792-4990-91b3-67394be380b9.jpg)

![giraffe](https://user-images.githubusercontent.com/81680367/146063048-0e300c53-ed54-41c7-97ef-dc079142c423.jpg)

**Note**:
* TinyYoloV2 trained by pascal VOC can not predict above image

* The bounding boxes and class prediction of YoloV2 for mentioned examples is less accurate

## Info

for more information about YoloV2, TinyYoloV2 implementations you can visit

https://github.com/miladlink/YoloV2

https://github.com/miladlink/TinyYoloV2