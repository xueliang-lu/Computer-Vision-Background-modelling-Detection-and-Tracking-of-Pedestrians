# Computer-Vision-Background-modelling-Detection-and-Tracking-of-Pedestrians
Computer Vision Algorithms and Systems

Objectives

Design a Python program that extracts and counts moving objects, e.g. people, cars and others using background modelling and detects pedestrians using pre-trained MobileNet SSD object detector.

Introduction

Extraction of moving objects and detection of pedestrians from a sequence of images or video is often used in many video analysis tasks. For instance, it is a key component in intelligent video surveillance systems and autonomous driving. In this assignment, you are required to develop a program in Python using OpenCV 4.6.0 to detect, separate and count moving objects from a given sequence of images or video captured by a stationary camera and to detect pedestrians. There are two tasks.

Task One – Background modelling
In this task, you are required to extract moving objects using Gaussian Mixture background modelling. There are three key steps involved in the extracting and counting moving objects:
1. Resize the video frame to a size comparable to VGA
2. Detecting moving pixels using background modelling and subtraction,
3. Removing noisy detection using morphological operators or majority voting and
4. Count separate moving objects using connected component analysis.
5. Classify each object (or connected component) into person, car and other by simply using the ratio of width and height of the connected components.
   
OpenCV 4.6.0 provides various algorithms for each of the steps 1 to 3. However, you MAY have to implement your own connected component analysis algorithm and classification algorithm. For simplicity, you can assume that each connected component corresponds to one object.

When running, the program should display the original video frame, estimated background frame, detected moving pixels after the background modeling and subtraction (before any noise removal) and the detected moving objects in a single window. The detected moving pixels before filtering should be displayed in black and white (binary mask). The detected objects have to be displayed in its original RGB color (all background pixels should be displayed in black). At the same time, the number of objects or connected components should be output to the command window as illustrated below:
  
   Frame 0001: 0 objects
   
   Frame 0002: 0 objects
   
   ...
   
   Frame 0031: 5 objects (2 persons, 1 car and 2 others)
   
   Frame 0032: 6 objects (3 persons, 1 cars and 2 others)
   
   ...
   
   Frame 1000: 10 objects (
   
   ...

Task Two – Detection and Tracking of Pedestrians

In this task, you are required to detect pedestrians (i.e. persons) using a OpenCV Deep Neural Network (DNN) module and a MobileNet SSD detector pre-trained on the MS COCO dataset, to track and display the detected pedestrians by providing same labels, i.e. 1, 2, ⋯ , 𝑛, to the same pedestrians across over times and to select up to (3) pedestrians that are most close in space to the camera. Solutions to the tracking and selection of up to three (3) most close pedestrians must be described in the comments at the beginning of your Python code.

Note that

(a) the pre-trained MobileNet SSD is provided:

• model configuration – ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt

• weights – frozen_inference_graph.pb

• names of object classes in MS COCO - object_detection_classes_coco.txt

(b) the size of the input images to the model is 300 × 300

(c) a quick tutorial on how to use the OpenCV DNN module and the pre-trained MobileNet SSD is
available at https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/.

(d) The pre-trained MobileNet SSD model is able to detect 80 classes of objects. This assignment is only
interested in pedestrians, i.e. persons.

The program should display the original video frame, video frame with overlapped bounding-boxes of the detected pedestrians, video frame with detected and tracked bounding-boxes and video frame with up to three (3) closest objects to the camera. Display must be in a single window.

Requirements on coding

The program should be named as “movingObj” and shall take an option, either –b or -d and a video filename as the input, e.g. movingObj –b videofile or movingObj –d videofile. When –b is given, the program should perform Task One and when –d is given, the program performs Task Two.
