delTri:
Image on face:
python3 Wrapper.py --method delTri --DataPath ../Data/TestSet/ --VideoFileName Test1.mp4 --RefImageFileName Rambo.jpg --SavePath ../Results/ --PredictorPath ./phase1/Misc/shape_predictor_68_face_landmarks.dat --use_filter True

python3 Wrapper.py --method delTri --DataPath ../Data/TestSet/ --VideoFileName Test3.mp4 --RefImageFileName Scarlett.jpg --SavePath ../Results/ --PredictorPath ./phase1/Misc/shape_predictor_68_face_landmarks.dat --use_filter False

Swap faces within a video:
python3 Wrapper.py --method delTri --DataPath ../Data/TestSet/ --VideoFileName Test2.mp4  --SavePath ../Results/


TPS:
Image on face:
python3 Wrapper.py --method TPS --DataPath ../Data/TestSet/ --VideoFileName Test1.mp4 --RefImageFileName Rambo.jpg --SavePath ../Results/ --PredictorPath ./phase1/Misc/shape_predictor_68_face_landmarks.dat --use_filter True

python3 Wrapper.py --method TPS --DataPath ../Data/TestSet/ --VideoFileName Test3.mp4 --RefImageFileName Scarlett.jpg --SavePath ../Results/ --PredictorPath ./phase1/Misc/shape_predictor_68_face_landmarks.dat --use_filter False

Swap faces within a video:
python3 Wrapper.py --method TPS --DataPath ../Data/TestSet/ --VideoFileName Test2.mp4  --SavePath ../Results/


PRNet:
Image on face:
python3 Wrapper.py --method PRNet --DataPath ../Data/TestSet/ --VideoFileName Test1.mp4 --RefImageFileName Rambo.jpg --SavePath ../Results/

python3 Wrapper.py --method PRNet --DataPath ../Data/TestSet/ --VideoFileName Test3.mp4 --RefImageFileName Scarlett.jpg --SavePath ../Results/


Swap faces within a video:
python3 Wrapper.py --method PRNet --DataPath ../Data/TestSet/ --VideoFileName Test2.mp4  --SavePath ../Results/


