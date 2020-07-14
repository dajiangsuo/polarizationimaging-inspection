import cv2
import numpy as np

file_image0 = 'Saran_0.jpg'
file_image45 = 'Saran_45.jpg'
file_image90 = 'Saran_90.jpg'

#reading images in bgr format
image0 = cv2.cvtColor(cv2.imread(file_image0),cv2.COLOR_BGR2GRAY)
image45 = cv2.cvtColor(cv2.imread(file_image45),cv2.COLOR_BGR2GRAY)
image90 = cv2.cvtColor(cv2.imread(file_image90),cv2.COLOR_BGR2GRAY)

#calculating stokes parameters
I = image0 + image90 + 0.01
Q = image0 - image90
U = 2*image45 - I
Intensity = np.sqrt(np.square(Q) + np.square(U))
DoLP = Intensity/I
AoP = 0.5*np.arctan2(U,Q)

#Use polarization info to convert to HSV
H = np.uint8(AoP*(180/np.pi))
S = np.uint8(255*(DoLP/np.amax(DoLP)))
V = np.uint8(255*(Intensity/np.amax(Intensity)))
HSV = cv2.merge((H,S,V))
new_image = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)

height, width = new_image.shape[:2]
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', width, height)
cv2.imshow('image', new_image)
cv2.waitKey(0)
