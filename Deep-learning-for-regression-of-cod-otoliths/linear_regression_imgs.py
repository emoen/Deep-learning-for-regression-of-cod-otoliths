from scipy.spatial import distance as dist
import scipy.misc
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import scipy.ndimage
 
#####################
image = cv2.imread("IMG_0067.JPG", cv2.IMREAD_UNCHANGED);
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray = cv2.GaussianBlur(gray, (7, 7), 0)

edges = cv2.Canny(gray, 20, 80)
scipy.misc.imsave('edge5.jpg', edges)

edges2 = scipy.ndimage.binary_dilation(edges).astype(edges.dtype)
edges2 = scipy.ndimage.binary_erosion(edges).astype(edges.dtype)
scipy.misc.imsave('edge6.jpg', edges2)

ret, thresh = cv2.threshold(edges, 127, 255, 0)
contour, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
d = cv2.drawContours(gray, contours, -1, (0,255,0), 3)
scipy.misc.imsave('d.jpg', d)

binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

scipy.misc.imsave('outfile.jpg', gray)

#

image = cv2.imread('IMG_0067.JPG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contour, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
count2 = cv.drawContours(im, contours, -1, (0,255,0), 3)

cv2.drawContours(image = gray, 
    contours = contours, 
    contourIdx = -1, 
    color = (0, 0, 255), 
    thickness = 5)

# 1) find countours
des = cv2.bitwise_not(gray)
contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contour:
    cv2.drawContours(des,[cnt],0,255,-1)

countoured = cv2.bitwise_not(des)
scipy.misc.imsave('contoured.jpg', countoured)

# 2) morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
res = cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel)
scipy.misc.imsave('morphed.jpg', res)

# getting mask with connectComponents
ret, labels = cv2.connectedComponents(binary)
mask = np.array(labels, dtype=np.uint8)
mask[labels == labels[0]] = 255
mask2 = scipy.ndimage.binary_fill_holes(mask).astype(int)
scipy.misc.imsave('mask2.jpg', mask2)


for label in range(1,ret):
    mask = np.array(labels, dtype=np.uint8)
    mask[labels == label] = 255
    cv2.imshow('component',mask)
    cv2.waitKey(0)

# getting ROIs with findContours
contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
for cnt in contours:
    (x,y,w,h) = cv2.boundingRect(cnt)
    ROI = image[y:y+h,x:x+w]
    cv2.imshow('ROI', ROI)
    cv2.waitKey(0)

cv2.destroyAllWindows()
#########################

_, lab = cv2.connectedComponents(gray)

#4-way connected pixels are neighbors to every pixel that touches one 
#of their edges. 8-way connected pixels are neighbors to every pixel that 
#touches one of their edges or corners. 
#This will help: aishack.in/tutorials/pixel-neighbourhoods-connectedness
# Choose 4 or 8 for connectivity type
connectivity = 4 
output = cv2.connectedComponentsWithStats(gray, connectivity, cv2.CV_32S)

num_labels = output[0]
stats = output[2]

for label in range(1,num_labels):
    blob_area = stats[label, cv2.CC_STAT_AREA]
    blob_width = stats[label, cv2.CC_STAT_WIDTH]
    blob_height = stats[label, cv2.CC_STAT_HEIGHT]
