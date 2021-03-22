import cv2
from glob import glob
import numpy as np
import pandas as pd
from time import time
import ast
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import imutils
import random
import math
import os
import dlib
from scipy.interpolate import interp2d

from phase1.Misc.utils import *
from phase1.dTriangulation import *
from phase1.ThinPlateSpline import *

from phase2.api_ import PRN_
from phase2.api import PRN
from phase2.prnet import *

from utils.render import render_texture
from utils.MovingAverage import *



print("###################### Done Imports ######################")

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--method', default="delTri", help='type of Faceswapper')
    Parser.add_argument('--DataPath', default="../Data/TestSet/", help='base path where data files exist')
    Parser.add_argument('--VideoFileName', default="Test1.mp4", help='Video Name')
    Parser.add_argument('--RefImageFileName', default='NONE', help=' Reference Image')
    Parser.add_argument('--SavePath', default="../Results/", help='Folder to save results')
    Parser.add_argument('--PredictorPath', default='./phase1/Misc/shape_predictor_68_face_landmarks.dat', help= 'dlib shape predictor path')
    Parser.add_argument('--use_filter', default= False, type = lambda x: bool(int(x)), help='use filter or not')

    Args = Parser.parse_args()
    DataPath = Args.DataPath
    RefImageFileName = Args.RefImageFileName
    SavePath = Args.SavePath
    method = Args.method
    VideoFileName = Args.VideoFileName
    path_to_shape_predictor = Args.PredictorPath
    use_filter = Args.use_filter
    
    RefImageFilePath = DataPath + RefImageFileName
    VideoFilePath = DataPath + VideoFileName
#     SavePath = SavePath + method + "/"      
    SaveFileName = DataPath + SavePath 

#     if(not (os.path.isdir(SavePath))):
#         print(SavePath, "  was not present, creating the folder...")
#         os.makedirs(SavePath) 

    #######################################
    ##### Setting up reference image ######
    ########################################
    print('Setting up reference image.......')
    FaceRef = cv2.imread(RefImageFilePath) ## color image
 
    ########## Choose Mode ############
    #######################################
    if FaceRef is None:
        mode = 2
    else:
        mode = 1
    print("we are in mode ", mode, " using ", method)

    ##### Setting up video ######
    ##############################
    cap = cv2.VideoCapture(VideoFilePath)
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))
    result = cv2.VideoWriter(SaveFileName,  
                            cv2.VideoWriter_fourcc(*'DIVX'), 
                            10, (frame_width, frame_height)) 

    ##############################################################################
    ########################### mode 1 ###########################################
    ##############################################################################
    if mode == 1:
        print("Doing Mode 1 with Reference Image")

        ##################################################
        ########### Setup pretrained models ##############
        ##################################################
        if method != 'PRNet':
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(path_to_shape_predictor)

            grayFaceRef = cv2.cvtColor(FaceRef, cv2.COLOR_BGR2GRAY) ## gray image
            N_pyrLayers = 1
            boxRef = detector(grayFaceRef,N_pyrLayers) ## reference bounding box
            if len(boxRef)!=1:
                print('Cannot find face in reference Image... exiting... :( ', len(boxRef))
                exit()
            else:
                boxRef = boxRef[0]
            ##ref image features
            FaceMarks_ref, BoundingBox_ref, shifted_FaceMarks_ref = FaceDetector(FaceRef, boxRef, predictor)
            (x,y,w,h) = BoundingBox_ref
            FaceRef_crop  = FaceRef[y:y+h,x:x+w]

        elif method == 'PRNet':
            prn = PRN(is_dlib = True)    


        ##################################################
        ################# Run the Loop  ##################
        ##################################################    
        facemarksFrames = []
        prev_pos = None
        #for filtering
        old_boxFrame = None
        del_boxFrame = 0
        first_time = True

        moving_average_position = MovingAverage(window_size = 4, weight = [1, 1, 1, 1])
        moving_average_del = MovingAverage(window_size = 4, weight = [1, 1, 1, 1])
        FaceMarkersFrame = 0
        while(True):    
            ret, frame = cap.read()
            if not ret:
                print("Stream ended..")
                break
            frame = adjustGamma(frame, 1.5)
            frame[frame > 255] = 255

            # frame = cv2.resize(frame, (ref_w,ref_h))

            ################# DeepLearning Method  ##################
            #########################################################
            if method == 'PRNet':

                pos = prn.process(frame)
                ref_pos = prn.process(FaceRef)

                if pos is None:
                    if prev_pos is not None:
                        pos = prev_pos
                    else:
                        print("## No face Found ##")
                        WarpedFace = frame
        #                 text_location = (frame_width-int(frame_width*0.2), frame_height- frame_height*0.1)
        #                 WarpedFace = cv2.putText(WarpedFace, 'Cannot Find Face',text_location, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0) ,3, cv2.LINE_AA, False)
                if pos is not None:
        #             if len(pos)>1:
        #                 print('Wrong Mode chosen')
        #                 exit()
        #             elif len(pos)==1:
                    WarpedFace = FaceSwap_DL(prn, pos, ref_pos, frame, FaceRef)

            #####################################################################
            ######################## Traditional methods ########################
            else:
                boxFrame = Top1Box(detector(gray(frame), 1))
                print(len(boxFrame))
                
                ### Failure Cases ###
                #Check if face is detected
                if len(boxFrame)>1:
                    margin = 50
                    boxRef,boxFrame  = boxFrame[0], boxFrame[1]
                    FaceFrame = FaceFrame[rects[1].top()-margin:rects[1].bottom()+margin,rects[1].left()-margin:rects[1].right()+margin,:]
                    FaceRef = FaceRef[rects[0].top()-margin:rects[0].bottom()+margin,rects[0].left()-margin:rects[0].right()+margin,:]

                elif(len(boxFrame) < 1):
                    if not first_time:
                        print("no box detected, using old box")
                        # boxFrame.append(old_boxFrame[0] + del_boxFrame)
                        if del_box_corners is not None:
                            boxFrame = predictBoxes(old_boxFrame, del_box_corners)
                        else:
                            boxFrame = old_boxFrame
                        
                    else:
                        print("No face found")
                        WarpedFace = frame 
                ########################################

                ### Success Cases ###
                if (len(boxFrame) > 0): 
                    # if not first_time:                
                    #     del_boxFrame = boxFrame[0] - old_boxFrame[0]
                    #     del_flag = True
                    # del_box = getDelBox(boxFrame, old_boxFrame)
                    del_box_corners =getDelBox(boxFrame, old_boxFrame)
                    old_boxFrame = boxFrame       
                    boxFrame = boxFrame[0]            

                    if method == 'TPS':

                        WarpedFace, FaceMarkersFrame = FaceSwap1_TPS(FaceRef_crop, shifted_FaceMarks_ref, frame, boxFrame, predictor, use_filter, first_time, FaceMarkersFrame, moving_average_position, moving_average_del)
                        #frame smoothening
                        if use_filter:
                            if not first_time:
                                # WarpedFace, del_frame = smoothenFrames(WarpedFace, past_frame)
                                past_frame = WarpedFace
                            else:
                                past_frame = WarpedFace

                        first_time = False

                    elif method == 'delTri':

                        WarpedFace, FaceMarkersFrame = FaceSwap1_delTri(FaceRef, FaceMarks_ref, frame, boxFrame, predictor, use_filter, first_time, FaceMarkersFrame, moving_average_position, moving_average_del)
                        #frame smoothening
                        if use_filter:
                            if not first_time:
                                WarpedFace, del_frame = smoothenFrames(WarpedFace, past_frame)
                                past_frame = WarpedFace
                            else:
                                past_frame = WarpedFace

                        first_time = False

                #######################
            cv2.imshow(str(method), WarpedFace)
            result.write(WarpedFace)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # MODE2_
    else:
        ##################################################
        ########### Setup pretrained models ##############
        ##################################################
        print("Doing Mode 2 Swap within frame")
        print(method)
        if method == 'PRNet':
            prn = PRN_(is_dlib = True)       
        else:
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(path_to_shape_predictor)

        ##################################################
        ################# Setup Video ####################
        ##################################################
        # print("setting up the video")
        # cap = cv2.VideoCapture(VideoFilePath)
        # frame_width = int(cap.get(3)) 
        # frame_height = int(cap.get(4))
        # result = cv2.VideoWriter(SaveFileName,  
        #                         cv2.VideoWriter_fourcc(*'DIVX'), 
        #                         10, (frame_width, frame_height))     

        facemarksFrames = []
        ##################################################
        ################ Run loop ########################
        ##################################################
        outputs = []
        print("Begin swapping")
        while(True):    
            ret, frame = cap.read()

            if not ret:
                print("Stream ended..")
                break

            if method == 'PRNet':

                poses = prn.process(frame)
                if poses is None:
                    poses = prev_poses
                print("number of Faces found...", len(poses))
                if len(poses)<2:
                    poses = prev_poses
                if len(poses) == 2:
                    prev_poses = poses
                    pose1 ,pose2 = poses[0],poses[1]
                    WarpedFace_tmp = FaceSwap_DL(prn, pose1, pose2, frame, frame)
                    WarpedFace = FaceSwap_DL(prn, pose2, pose1, WarpedFace_tmp, frame)
                else:
                    WarpedFace = frame
                    text_location = (frame_width-400, frame_height-42)
                    WarpedFace = cv2.putText(WarpedFace, 'Cannot Find Face',text_location, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0) ,3, cv2.LINE_AA, False)        

            else:
                boxes = Top2Boxes(detector(gray(frame), 1))
                print("number of found...", len(boxes))

                # if len(boxes)<2:
                #     boxes = old_box   
                if boxes is None or len(boxes) < 2:
                    if not first_time:
                        boxes = old_box


                if boxes is not None and len(boxes) ==2: #review
                    old_box = boxes
                    # prev_boxes = boxes
                    box1, box2 = boxes[0], boxes[1]
                    if method == 'delTri':
                        WarpedFace = FaceSwap2_delTri(frame,boxes[0], boxes[1], predictor)
                        first_time = False
                    elif method == 'TPS':
                        WarpedFace = FaceSwap2_TPS(frame,boxes[0], boxes[1], predictor)
                        first_time = False
                else:
                    WarpedFace = frame
                    text_location = (frame_width-400, frame_height-42)
                    WarpedFace = cv2.putText(WarpedFace, 'Cannot Find Face',text_location, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0) ,3, cv2.LINE_AA, False)        

            cv2.imshow(str(method), WarpedFace)
            result.write(np.uint8(WarpedFace))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    result.release() 
    cv2.destroyAllWindows()
    
    
    
# %%
if __name__ == '__main__':
    main()
