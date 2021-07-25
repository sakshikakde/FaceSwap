
# FaceSwap
## Overview
### PHASE 1: TRADITIONAL APPROACH
![alt test](https://github.com/sakshikakde/FaceSwap/blob/master/git_images/Screenshot%20from%202021-07-25%2013-44-36.png)
#### 1. Facial Landmarks Detection
To perform Face swap, we need to first detect a human
face in a given image frame. In order to detect a human
face, we utilise a Histogram of Oriented Gradients (hog)
+ Support Vector machine (svm) based face detector, that
is built in the dlib library
![alt text](https://github.com/sakshikakde/FaceSwap/blob/master/git_images/Screenshot%20from%202021-07-25%2013-43-50.png)
#### 2. Face Warping 
##### Using Triangulation
![alt text](https://github.com/sakshikakde/FaceSwap/blob/master/git_images/Screenshot%20from%202021-07-25%2013-45-14.png)
##### Using Thin Plate Spline
![alt text](https://github.com/sakshikakde/FaceSwap/blob/master/git_images/Screenshot%20from%202021-07-25%2013-45-01.png)
#### 3. Blending
The source face, once reshaped using one of the above
warping methods, were blended in the location of the destination image frame using Poisson blending. To perform Poisson
blending, we need compute a mask of the face in destination
image, where the region of interest (ROI) of the face has white
pixels and the rest of the frame is black.
![alt text](https://github.com/sakshikakde/FaceSwap/blob/master/git_images/Screenshot%20from%202021-07-25%2013-44-49.png)
#### 4. Motion Filtering
### PHASE 2: DEEP LEARNING APPROACH
## Run the code
1. Download the Face Landmarks detector file from [here](dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), unzip and copy it to Misc/ Folder
2. Download the PRNet pretrained model file from [here](https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view), unzip and copy it to phase2/Data/net-data/ Folder
3. Copy the dlib facial marker detector model to Code/phase1/Misc/
4. For PRNet, copy the pretrained model file to Code/phase2/Data/net-data/ 
5. To run the program    
  ```cd Code/```.

### 1. delTri:
#### Image on face:
```
python3 Wrapper.py --method delTri --DataPath ../Data/TestSet/ --VideoFileName Test1.mp4 --RefImageFileName Rambo.jpg --SavePath Test1OutputTri.mp4 --PredictorPath ./phase1/Misc/shape_predictor_68_face_landmarks.dat --use_filter 1
```
```
python3 Wrapper.py --method delTri --DataPath ../Data/TestSet/ --VideoFileName Test3.mp4 --RefImageFileName Scarlett.jpg --SavePath Test3OutputTri.mp4 --PredictorPath ./phase1/Misc/shape_predictor_68_face_landmarks.dat --use_filter 0
```
#### Swap faces within a video:
python3 Wrapper.py --method delTri --DataPath ../Data/TestSet/ --VideoFileName Test2.mp4  --SavePath Test2OutputTri.mp4


### 2. TPS:
#### Image on face:
```
python3 Wrapper.py --method TPS --DataPath ../Data/TestSet/ --VideoFileName Test1.mp4 --RefImageFileName Rambo.jpg --SavePath Test1OutputTPS.mp4 --PredictorPath ./phase1/Misc/shape_predictor_68_face_landmarks.dat --use_filter 1
```
```
python3 Wrapper.py --method TPS --DataPath ../Data/TestSet/ --VideoFileName Test3.mp4 --RefImageFileName Scarlett.jpg --SavePath Test3OutputTPS.mp4 --PredictorPath ./phase1/Misc/shape_predictor_68_face_landmarks.dat --use_filter 0
```
#### Swap faces within a video:
```
python3 Wrapper.py --method TPS --DataPath ../Data/TestSet/ --VideoFileName Test2.mp4  --SavePath Test2OutputTPS.mp4
```

### 3.PRNet:
#### Image on face:
```
python3 Wrapper.py --method PRNet --DataPath ../Data/TestSet/ --VideoFileName Test1.mp4 --RefImageFileName Rambo.jpg --SavePath Test1OutputPRNet.mp4 --PredictorPath ./phase1/Misc/shape_predictor_68_face_landmarks.dat --use_filter 1
```
```
python3 Wrapper.py --method PRNet --DataPath ../Data/TestSet/ --VideoFileName Test3.mp4 --RefImageFileName Scarlett.jpg --SavePath Test3OutputPRNet.mp4 --PredictorPath ./phase1/Misc/shape_predictor_68_face_landmarks.dat --use_filter 0
```
#### Swap faces within a video:
```
python3 Wrapper.py --method PRNet --DataPath ../Data/TestSet/ --VideoFileName Test2.mp4  --SavePath Test2OutputPRNet.mp4
```

