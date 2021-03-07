import cv2
import dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def readImages(path1,path2):
    Face1 = cv2.imread(path1)
    Face2 = cv2.imread(path2)
    h_min = min(Face1.shape[0], Face2.shape[0])
    w_min = min(Face1.shape[1], Face2.shape[1])
    Face1 = cv2.resize(Face1, (w_min, h_min))
    Face2 = cv2.resize(Face2, (w_min, h_min))
    
    Face1 = rescale(Face1,scale_percent = 50)
    Face2 = rescale(Face2,scale_percent = 50)
    return Face1, Face2

def rescale(src,scale_percent = 50):
    #calculate the x percent of original dimensions
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    return cv2.resize(src, (width, height))

def getcv2Box(rect):
    """
    Take a bounding predicted by dlib and convert it to the format (x, y, w, h) 
    
    """
    
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def toNumpy(shape, dtype="int"):
    """
    Loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y) numpy arrays
    """
    
    # initialize the list of (x, y)-coordinates
    pts = np.zeros((68, 2), dtype=dtype)
    
    for i in range(0, 68):
        pts[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return pts

# Check if a point is inside a rectangle
def isPointInside(pt, box) :
    x,y =  pt
    x_min,y_min,width,height = box
    x_max,y_max = x_min + width, y_min + height
    
    if (x > x_min) and (x < x_max) and (y > y_min) and (y < y_max):
        return True
    else:
        return False
    
def PlotaTri(Face_, tri, colour = (0,255,255)):
    Face = Face_.copy()
    a1,b1,c1 = tuple(tri[0]), tuple(tri[1]), tuple(tri[2])
    
    Face = cv2.line(Face, a1, b1, colour, 2)
    Face = cv2.line(Face, b1, c1, colour, 2)
    Face = cv2.line(Face, c1, a1, colour, 2)    
    return Face