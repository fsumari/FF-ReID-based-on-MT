# Full-Frame person Re-identification

* This the code for security-level of person Re-identification on real scenarios.

## part 1. Quick start
1. Clone this file
```bashrc
$ git clone https://github.com/fsumari/FF-ReID-based-on-MT.git
```
2.  The code is compatible with Python 3.7. The following dependencies are needed to run the FF-PRID:

```bashrc
Tensorflow 1.4.0
Keras
scikitlearn
NumPy
OpenCV
pandas
matplotlib
```
# 1.1 yolov3 in TF 1.14.0

* Donwload model of yolov3 for tf and save on directory 'pYOLO/checkpoint'
* link of model is below:

link: https://drive.google.com/file/d/1kQ6BTs3-B6A_fUUDqsYFGp7yYL0d2k87/view?usp=sharing 

* also, the yolov3_coco.pb donwload and save in '/pYOLO'

link: https://drive.google.com/open?id=1WxpW_UB6ci4vC_34CnpvXKoAaqz8DiOF

# 2. Execution


2.1. Exporting loaded Re-id weights as TF checkpoint(`model.ckpt`)
```bashrc
$ cd pReID/logs
$ wget https://drive.google.com/file/d/1pFAIkjLrNB0KeKLWuUlS0hoJIwXR00oV/view?usp=sharing
$ unzip weightsReid.zip
$ cd ..
```
2.2. Run the simple test for FF-PRID
* Execute test for FF-PRID only with YOLOv3 like to cropper:
```
   python demo.py --mode=rw_test --query_path=querys/person_0015.png --video_path=data/seq1/video_in.avi
  ```
* Execute test for FF-PRID with YOLOv3 + tracker like to cropper:
```
   python demo.py --mode=mt_test --query_path=querys/person_0015.png --video_path=data/seq1/video_in.avi
  ```

