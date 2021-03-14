import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import imutils
import random
import math

from phase1.Misc.utils import *
from scipy.interpolate import interp2d
import os
import dlib



def ThinPlateKernalFunc(set1, set2):
    # r = np.abs(set1[0] -  set2[0]) + np.abs(set1[1] - set2[1])
    r = np.linalg.norm((np.array(set1) - np.array(set2)))
    U = (r**2) * np.log(r**2)
    if r == 0:
        U = 0
    return U

def getParameters(set1, set2):
    p = set2.shape[0]
    K = np.zeros((p, p))
    P = np.zeros((p, 3))

    for i in range(p):
        for j in range(p):
            K[i,j] = ThinPlateKernalFunc([set2[i, 0], set2[i, 1]], [set2[j, 0], set2[j, 1]])

        P[i, 0] = set2[i, 0]
        P[i, 1] = set2[i, 1]
        P[i, 2] = 1

    L = np.zeros((p + 3, p + 3))
    L[0:p, 0:p] = K
    L[0:p, p:p + 3] = P
    L[p:p + 3, 0:p] = np.transpose(P)

    L = L + np.eye(p + 3) * 1e-10
    L_inv = np.linalg.inv(L)

    set1x_dash = np.zeros((set1.shape[0] + 3, 1))
    set1y_dash = np.zeros((set1.shape[0] + 3, 1))

    set1x_dash[0:set1.shape[0], :] = set1[:,0].reshape(-1,1)
    set1y_dash[0:set1.shape[0], :] = set1[:,1].reshape(-1,1)

    Mx = np.dot(L_inv, set1x_dash)
    My = np.dot(L_inv, set1y_dash)

    return np.hstack((Mx, My))


def warpFaces_TPS(face1, face2, M, face_markers_2):
    Mx = M[:, 0].reshape(-1,1)
    My = M[:, 1].reshape(-1,1)

    Xi, Yi = np.indices((face2.shape[1], face2.shape[0])) 
    warped_points = np.stack((Xi.ravel(), Yi.ravel(), np.ones(Xi.size))).T

    axx = Mx[Mx.shape[0] - 3]
    ayx = Mx[Mx.shape[0] - 2]
    a1x = Mx[Mx.shape[0] - 1]

    axy = My[My.shape[0] - 3]
    ayy = My[My.shape[0] - 2]
    a1y = My[My.shape[0] - 1]

    A = np.array([[axx, axy], [ayx, ayy], [a1x, a1y]]).reshape(3,2)
    actual_points = np.dot(warped_points, A) 

    warped_points_x = warped_points[:,0].reshape(-1,1)
    face_markers_2_x = face_markers_2[:,0].reshape(-1,1)
    ax, bx = np.meshgrid(face_markers_2_x, warped_points_x)
    t1 = np.square(ax - bx)

    warped_points_y = warped_points[:, 1].reshape(-1,1)
    face_markers_2_y = face_markers_2[:, 1].reshape(-1,1)
    ay, by = np.meshgrid(face_markers_2_y, warped_points_y)
    t2 = np.square(ay - by)

    R = np.sqrt(t1 + t2)
    U = np.square(R) * np.log(np.square(R))
    U[R == 0] = 0 #perfect"

    MX = Mx[0:68, 0].T
    Ux = MX * U #perfect
    Ux_sum = np.sum(Ux, axis = 1).reshape(-1,1)

    MY = My[0:68, 0].T
    Uy = MY * U #perfect
    Uy_sum = np.sum(Uy, axis = 1).reshape(-1,1)

    actual_points = actual_points + np.hstack((Ux_sum, Uy_sum))

    X = actual_points[:, 0].astype(int)
    Y = actual_points[:, 1].astype(int)
    X[X >= face1.shape[1]] = face1.shape[1] - 1
    Y[Y >= face1.shape[0]] = face1.shape[0] - 1
    X[X < 0] = 0
    Y[Y < 0] = 0

    warped_face = np.zeros(face2.shape)
#     print(np.max(Yi.ravel()), np.max(Xi.ravel()))
#     print(np.max(Y.ravel()), np.max(X.ravel()))
    warped_face[Yi.ravel(), Xi.ravel()] = face1[Y, X]

    return np.uint8(warped_face)

def FaceSwap_TPS(Face1,Face2, box1,box2, predictor):
    ## face1 - src; face2 - dst
    
    ##### Get the Face fiducials #####
    facemarks1, BoundingBox1, shifted_FaceMarks1 = FaceDetector(Face1, box1, predictor)
    facemarks2, BoundingBox2, shifted_FaceMarks2 = FaceDetector(Face2, box2, predictor)    
    (x,y,w,h) = BoundingBox1
    Face1_crop  = Face1[y:y+h,x:x+w]
    (x,y,w,h) = BoundingBox2
    Face2_crop  = Face2[y:y+h,x:x+w]
    #####------#####
    
    ##### Get the warpedFace using TPS ##### 
    M  = getParameters(shifted_FaceMarks1, shifted_FaceMarks2)
    warped_face = warpFaces_TPS(Face1_crop, Face2_crop, M, shifted_FaceMarks2)
    #####------#####

    ##### Move the warped face to destination location #####
    mask_warped_face,_ = Mask(warped_face, shifted_FaceMarks2)
    mask_warped_face = np.int32(mask_warped_face/mask_warped_face.max())
    warped_face = warped_face * mask_warped_face
    alpha = (1.0, 1.0, 1.0) - mask_warped_face
    
    WarpedFace = Face1.copy() # get a copy of dst face
    WarpedFace = np.zeros_like(Face1) # or just zeros in same shape
    x,y,w,h = BoundingBox2
    WarpedFace[y:y+h, x:x+w] = WarpedFace[y:y+h, x:x+w] * alpha
    WarpedFace[y:y+h, x:x+w] = WarpedFace[y:y+h, x:x+w] + warped_face
    #####------#####
    
    ### Perform Poisson Blending to paste the image ###
    mask, box = Mask(Face2,facemarks2) # get the mask of the face and it's bounding box 
    x,y,w,h = box  
    cx, cy = (2*x+w) //2, (2*y+ h) //2   # get the center
    WarpedFace = cv2.seamlessClone(np.uint16(WarpedFace), Face2, mask, tuple([cx,cy]), cv2.NORMAL_CLONE)
    #####------#####
    
    
    Face1_print = drawMarkers(Face1, facemarks1, BoundingBox1)
    Face2_print = drawMarkers(Face2, facemarks2, BoundingBox2)
    
    return WarpedFace, Face1_print, Face2_print


