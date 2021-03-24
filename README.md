# FaceSwap


Download the Face Landmarks detector file from [here](dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), unzip and copy it to Misc/ Folder


Download the PRNet pretrained model file from [here](https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view), unzip and copy it to phase2/Data/net-data/ Folder


Copy the dlib facial marker detector model to Code/phase1/Misc/

For PRNet, copy the 
Copy the pretrained model file to Code/phase2/Data/net-data/ 


To run the program cd Code/

# delTri:
## Image on face:
```
python3 Wrapper.py --method delTri --DataPath ../Data/TestSet/ --VideoFileName Test1.mp4 --RefImageFileName Rambo.jpg --SavePath Test1OutputTri.mp4 --PredictorPath ./phase1/Misc/shape_predictor_68_face_landmarks.dat --use_filter 1
```
```
python3 Wrapper.py --method delTri --DataPath ../Data/TestSet/ --VideoFileName Test3.mp4 --RefImageFileName Scarlett.jpg --SavePath Test3OutputTri.mp4 --PredictorPath ./phase1/Misc/shape_predictor_68_face_landmarks.dat --use_filter 0
```
## Swap faces within a video:
python3 Wrapper.py --method delTri --DataPath ../Data/TestSet/ --VideoFileName Test2.mp4  --SavePath Test2OutputTri.mp4


# TPS:
## Image on face:
```
python3 Wrapper.py --method TPS --DataPath ../Data/TestSet/ --VideoFileName Test1.mp4 --RefImageFileName Rambo.jpg --SavePath Test1OutputTPS.mp4 --PredictorPath ./phase1/Misc/shape_predictor_68_face_landmarks.dat --use_filter 1
```
```
python3 Wrapper.py --method TPS --DataPath ../Data/TestSet/ --VideoFileName Test3.mp4 --RefImageFileName Scarlett.jpg --SavePath Test3OutputTPS.mp4 --PredictorPath ./phase1/Misc/shape_predictor_68_face_landmarks.dat --use_filter 0
```
## Swap faces within a video:
```
python3 Wrapper.py --method TPS --DataPath ../Data/TestSet/ --VideoFileName Test2.mp4  --SavePath Test2OutputTPS.mp4
```

# PRNet:
## Image on face:
```
python3 Wrapper.py --method PRNet --DataPath ../Data/TestSet/ --VideoFileName Test1.mp4 --RefImageFileName Rambo.jpg --SavePath Test1OutputPRNet.mp4 --PredictorPath ./phase1/Misc/shape_predictor_68_face_landmarks.dat --use_filter 1
```
```
python3 Wrapper.py --method PRNet --DataPath ../Data/TestSet/ --VideoFileName Test3.mp4 --RefImageFileName Scarlett.jpg --SavePath Test3OutputPRNet.mp4 --PredictorPath ./phase1/Misc/shape_predictor_68_face_landmarks.dat --use_filter 0
```
## Swap faces within a video:
```
python3 Wrapper.py --method PRNet --DataPath ../Data/TestSet/ --VideoFileName Test2.mp4  --SavePath Test2OutputPRNet.mp4
```

