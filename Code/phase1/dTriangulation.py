import cv2
import numpy as np
import pandas as pd
from phase1.Misc.utils import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import Counter
import argparse
from scipy.spatial import distance as dist
import imutils
import random
import math
from skimage.feature import peak_local_max
from scipy.interpolate import interp2d
import os
import dlib


"""
Consists Delauney Triangulation Functions

"""

def WarpTriangles(Face1, warpedFace1 ,tri1,tri2, iscv2 = False):
    # get a rectangular rergion spanning the triangles.
    BBox1 = cv2.boundingRect(np.float32([tri1]))
    BBox2 = cv2.boundingRect(np.float32([tri2]))

    x1,y1,w1,h1 = BBox1
    x2,y2,w2,h2 = BBox2

    tri_shifted1 = [] 
    for t in tri1:
        tri_shifted1.append([(t[0] - x1 , t[1] - y1)])

    tri_shifted1 = np.array(tri_shifted1).astype(np.int32)

    tri_shifted2 = []    
    for t in tri2:
        tri_shifted2.append([(t[0] - x2 , t[1] - y2)])

    tri_shifted2 = np.array(tri_shifted2).astype(np.int32)

    # select the src image pixels within the box outscribing the triangle 1
    triBox1 = Face1[y1:y1+h1, x1:x1+w1]
    triBox2 = warpedFace1[y2:y2+h2, x2:x2+w2]
    
    triBox2 = affineTransform(triBox1, tri_shifted1, tri_shifted2, triBox2, iscv2)
    
    # make a mask of the dst triangle
    mask = np.zeros((h2, w2, 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, tri_shifted2, (1.0, 1.0, 1.0), 16, 0);

    triBox2 = triBox2 * mask
    alpha = (1.0, 1.0, 1.0) - mask
    # Copy triangular region of the rectangular patch to the output image
    warpedFace1[y2:y2+h2, x2:x2+w2] = warpedFace1[y2:y2+h2, x2:x2+w2] * alpha
    warpedFace1[y2:y2+h2, x2:x2+w2] = warpedFace1[y2:y2+h2, x2:x2+w2] + triBox2

def affineTransform(src, src_tri, dst_tri, dst, iscv2 = False):
    (h2,w2) = dst.shape[:2]
    if iscv2 == False:
#         dst = np.zeros((h2,w2,3), np.uint8)

        ## compute the B matrix that to compute Barycentric Coordinate
        ax,ay = dst_tri[0].squeeze()
        bx,by = dst_tri[1].squeeze()
        cx,cy = dst_tri[2].squeeze()

        B = np.array([[ax,bx,cx],
                      [ay,by,cy],
                      [1, 1, 1]])
        
        if np.linalg.det(B)!=0:
            B_inv = np.linalg.inv(B)
        else:    
            B_inv = np.linalg.pinv(B)
#             B_inv = np.linalg.lstsq(B, np.eye(3, 3))[0]

        ## create [x,y,1] -> the coordinates of the box which outscribe the triangle.
        dstBox = cv2.boundingRect(np.float32(dst_tri))
        x,y,w,h = dstBox

        X, Y  = np.mgrid[x:x+w, y:y+h].reshape(2,-1)
        ones = np.ones((1, Y.shape[0]))
        XY1 = np.vstack((X,Y,ones))

        ## get [alpha,beta,gamma] -> the Barycentric coordinates
        abg = B_inv.dot(XY1)

        ## Compute points where alpha, beta and gamma are within [0,1] - these are inliers, rest are outliers

        e = 0.01
        alpha, beta, gamma = abg[0],abg[1], abg[2]

        ## get all indices that fulfil the condition
        idxs = []
        for i in range(len(alpha)):
            if ((alpha[i]>0-e) and (alpha[i]<1+e)) and ((beta[i]>0-e) and (beta[i]<1+e)) and ((gamma[i]>0-e) and (gamma[i]<1+e)):
                tmp_sum = alpha[i]+beta[i]+ gamma[i]
                if (tmp_sum)>0-e and (tmp_sum)<1+e :
                    idxs.append(i)
        abg_inliers = abg.T[idxs]


        # find the INLIER coordinates of triangle in destination location
        xy_b = XY1.T[idxs]
        Xb, Yb, z = xy_b[:,0], xy_b[:,1], xy_b[:,2]
        Xb, Yb = Xb/z, Yb/z

        ax,ay = src_tri[0].squeeze()
        bx,by = src_tri[1].squeeze()
        cx,cy = src_tri[2].squeeze()
        A = np.array([[ax,bx,cx],
                     [ay,by,cy],
                     [1, 1, 1]])

        # find the INLIER coordinates of triangle in source location
        xyz_a = A.dot(abg_inliers.T)
        xyz_a = xyz_a.T
        Xa, Ya,z = xyz_a[:,0], xyz_a[:,1], xyz_a[:,2] 
        Xa, Ya = np.int32(Xa/z), np.int32(Ya/z)
        h1,w1 = src.shape[:2]
        for xb,yb,xa,ya in zip(Xb,Yb,Xa,Ya):
            if (ya>=0) and (ya<h1) and (xa>=0) and (xa<w1):
                dst[int(yb),int(xb)] = src[int(ya),int(xa)]
        
    else:
        dst = affineWarping(src, src_tri, dst_tri, (w2,h2))
    return dst    

def DelauneyTriangles(Face2, Facemarks, colour = (0,255,255)):
    
    """
    get a List of triangles in the getTriangleList from subdiv2d and choose only the delauney triangles
    """
    
    Face = Face2.copy()
    #Create an instance of Subdiv2D with image dimensions 
    h,w = Face.shape[:2]    
    Box = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(tuple(Box))


    # insert the points
    for point in Facemarks:
        subdiv.insert(tuple(point)) 

    triangleList = subdiv.getTriangleList()
    
    data  = {}
    for i,point in enumerate(Facemarks): 
        ki = tuple(point) 
        data[ki] = i
    
    delTri = []
    MarkerInd = []
    ## for every triangle in the list
    for triangle in triangleList:

        ## get the vertices of the triangle
        a = (triangle[0], triangle[1])
        b = (triangle[2], triangle[3])
        c = (triangle[4], triangle[5])  
        vertices = [a,b,c]
        
        ## check if the triangle is inside the boundary box (or the image)
        boundaryCondition = isPointInside(a,Box) and isPointInside(b,Box) and isPointInside(c,Box)
        if  boundaryCondition:
            indices = []
            TrianglePoints = []
            ## find the facial markers that have a distance below 1.0 to the given vertice 
            for v in vertices :
                index = data.get(tuple(v),False) 
                if index:
                    TrianglePoints.append(np.array(v))                        
                    indices.append(index)
                    
            ## if there are exactly 3 points inside, then that is a delauney triangle                                                   
            if len(indices)==3:
                
                ordered_idx = np.argsort(np.array(indices))
                indices = np.array(indices)
                indices = indices[ordered_idx]
                TrianglePoints = np.array(TrianglePoints)
                TrianglePoints = TrianglePoints[ordered_idx]
    
                delTri.append(TrianglePoints)
                MarkerInd.append(tuple(indices))
    
    return delTri,MarkerInd, Face 

def getMatchingTriangles(Face2, facemarks2, facemarks1):
    ## get the Delauney Triangles of the destinationface
    delTri2,  tri_indices2, _ = DelauneyTriangles(Face2, facemarks2)
    
    ## get the corresponding Triangles of the sourceface
    delTri1= []
    for i in range(0, len(tri_indices2)):
        t1 = []
        for j in range(0, 3):
            t1.append(facemarks1[tri_indices2[i][j]])
        delTri1.append(np.int32(t1))
    return np.array(delTri1) , np.array(delTri2)


def affineWarping(src, srcTri, dstTri, size) :
    
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

def FaceSwap1_delTri(Face1, facemarks1, Face2, box2, predictor, use_filter, first_time, old_markers, moving_average_position, moving_average_del):
    
    ##### Get the Face fiducials #####
    facemarks2, _, _ = FaceDetector(Face2, box2, predictor)  
    if use_filter:
        print("Using filter")
        moving_average_position.addMarkers(facemarks2)
        facemarks2_averaged = moving_average_position.getAverage()  
        facemarks2 = facemarks2_averaged

        if not first_time:
            del_markers = facemarks2 - old_markers
            del_markers[del_markers > 3] = 3
            del_markers[del_markers < -3] = -3

            moving_average_del.addMarkers(del_markers)
            del_markers = moving_average_del.getAverage()

            facemarks2 = (old_markers + del_markers).astype(int)

    ## find the common delauney triangles in both faces using tri_indices
    delTri1 , delTri2 = getMatchingTriangles(Face2, facemarks2, facemarks1)
    #####------#####

    ## warp all triangles from source to destination
    WarpedFace = Face2.copy()
    for tri1,tri2 in zip(delTri1,delTri2):
        WarpTriangles(Face1,WarpedFace, tri1,tri2, False)    
    #####------#####

    ### Perform Poisson Blending to paste the image ###
    mask, box = Mask(Face2,facemarks2) # get the mask of the face and it's bounding box 
    x,y,w,h = box  
    cx, cy = (2*x+w) //2, (2*y+ h) //2   # get the center
    WarpedFace = cv2.seamlessClone(np.uint16(WarpedFace), Face2, mask, tuple([cx,cy]), cv2.NORMAL_CLONE)
    #####------#####
    
    # Face1_print = drawMarkers(Face1, facemarks1, BoundingBox1)
    # Face2_print = drawMarkers(Face2, facemarks2, BoundingBox2)
    
    return WarpedFace, facemarks2


def FaceSwap2_delTri(frame, box1, box2, predictor):
    
    facemarks1, BoundingBox1, shifted_FaceMarks1 = FaceDetector(frame, box1, predictor)
    facemarks2, BoundingBox2, shifted_FaceMarks2 = FaceDetector(frame, box2, predictor)

    ### del Triangles ### 
    # find the common delauney triangles in both faces using tri_indices    
    delTri1 , delTri2 = getMatchingTriangles(frame, facemarks2, facemarks1)
        
    ## warp all triangles from source to destination
    WarpedFace = frame.copy()
    for tri1,tri2 in zip(delTri1,delTri2):
        WarpTriangles(frame,WarpedFace, tri1,tri2, False)
        WarpTriangles(frame,WarpedFace, tri2,tri1, False)
    #####------#####
    
    ### Perform Poisson Blending to paste the image ###
    mask, box = Mask(frame,facemarks2) # get the mask of the face and it's bounding box
    x,y,w,h = box  
    cx, cy = (2*x+w) //2, (2*y+ h) //2   # get the center
    WarpedFace_tmp = cv2.seamlessClone(np.uint16(WarpedFace), frame, mask, tuple([cx,cy]), cv2.NORMAL_CLONE)        

    mask, box = Mask(frame,facemarks1) # get the mask of the face and it's bounding box
    x,y,w,h = box  
    cx, cy = (2*x+w) //2, (2*y+ h) //2   # get the center
    WarpedFace = cv2.seamlessClone(np.uint16(WarpedFace), WarpedFace_tmp, mask, tuple([cx,cy]), cv2.NORMAL_CLONE)        
    
    return WarpedFace