import cv2
import dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def readImages(path1,path2):
    Face1 = cv2.imread(path1)
    Face2 = cv2.imread(path2)
    Face1 = cv2.cvtColor(Face1, cv2.COLOR_BGR2RGB)
    Face2 = cv2.cvtColor(Face2, cv2.COLOR_BGR2RGB)
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

def AngleofTri(triangle2):
    ab = triangle2[0] - triangle2[1]
    ab = ab[0]**2 + ab[1]**2

    bc = triangle2[0] - triangle2[1]
    bc = bc[0]**2 + bc[1]**2

    ca = triangle2[0] - triangle2[1]
    ca = ca[0]**2 + ca[1]**2

    
    bc_root = math.sqrt(bc);  
    ca_root = math.sqrt(ca);  
    ab_root = math.sqrt(ab);  

    alpha = math.acos((ca + ab - bc) /(2 * ca_root * ab_root))  
    beta = math.acos((bc + ab - ca) /(2 * bc_root * ab_root))  
    gamma = math.acos((bc + ca - ab) /(2 * bc_root * ca_root))  

    # Converting to degree  
    alpha = alpha * 180 / math.pi;  
    beta = beta * 180 / math.pi;  
    gamma = gamma * 180 / math.pi;
    
    return [int(alpha), int(beta), int(gamma)]

def order_points(pts):
    """
    reference : https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    """
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]

    (tl, bl) = leftMost
    br = xSorted[2, :]

    return np.array([tl, br, bl])

def drawMarkers(Face1, facemarks, box):    
    Face = Face1.copy()
    (x,y,w,h) = box
    cv2.rectangle(Face,(x,y),(x+w,y+h),(0,255,255),1)
    
    for (a,b) in facemarks:
        cv2.circle(Face,(a,b),1,(0,0,255),-1)
    return Face

def Mask(Face2,facemarks2):
    mask = np.zeros_like(Face2)
    hull = cv2.convexHull(np.array(facemarks2))
    mask = cv2.fillConvexPoly(mask,hull, (255, 255, 255)) 
    box = cv2.boundingRect(np.float32([hull.squeeze()]))
    return mask , box

def FaceDetector(Face, box, FacePredictor):
    # pass the image and the bounding box to get face landmarks
    grayFace = cv2.cvtColor(Face, cv2.COLOR_BGR2GRAY)
    facemarks = FacePredictor(grayFace,box)
    facemarks = toNumpy(facemarks)
    BoundingBox = getcv2Box(box)
    (x,y,w,h) = BoundingBox
    shifted_FaceMarks = facemarks - (x,y)                
    return facemarks, BoundingBox, shifted_FaceMarks