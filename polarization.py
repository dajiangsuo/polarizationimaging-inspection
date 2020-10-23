import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#function to display the different image types
#if edges is true show the edges for each image
def displayImages(Intensity, AoP, DoLP, HSVinBGR, edges):
    images = [Intensity, AoP, DoLP, HSVinBGR]
    if edges:
        display_images = []
        plot_titles = ['Intensity: edges', 'AoP: edges', 'DoLP: edges', 'HSV Image in BGR: edges']
        for img in images:
            display_images.append(cv2.Canny(img,100,200))
    else:
        display_images = images
        plot_titles = ['1-channel: Intensity', '1-channel: AoP', '1-channel: DoLP', 'HSV Image in BGR']
    rows = 2
    cols = 2
    plt.figure("Polarization Parameters")
    for i in range(rows*cols):
        img = display_images[i]
        subplot_title = plot_titles[i]
        plt.subplot(2,2,i+1)
        plt.imshow(img)
        plt.title(subplot_title)
    plt.show()

#put folder name containing the three polarized images of desired object
folder_name = 'vial7_new/'
entries = os.listdir(folder_name)
object_name = entries[0].split('_')[0]
file_image0 = os.path.join(folder_name, entries[0])
file_image45 = os.path.join(folder_name, entries[1])
file_image90 = os.path.join(folder_name, entries[2])
file_imageNF = os.path.join(folder_name, entries[3])

#reading images in bgr format and converting to grayscale
image0 = np.uint64(cv2.cvtColor(cv2.imread(file_image0),cv2.COLOR_BGR2GRAY))
image45 = np.uint64(cv2.cvtColor(cv2.imread(file_image45),cv2.COLOR_BGR2GRAY))
image90 = np.uint64(cv2.cvtColor(cv2.imread(file_image90),cv2.COLOR_BGR2GRAY))

#calculating stokes parameters
I = image0 + image90
Q = image0 - image90
U = 2*image45 - image0 - image90
Intensity = np.sqrt(np.square(Q) + np.square(U))
DoLP = Intensity/I
AoP = 0.5*np.arctan2(U,Q)

#Use polarization info to convert to HSV
H = np.uint8((AoP+np.pi/2)*(180/np.pi))
S = np.uint8(255*(DoLP/np.amax(DoLP)))
V = np.uint8(255*(Intensity/np.amax(Intensity)))
HSV = cv2.merge((H,S,V))
HSVinBGR = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)

#Extract individual channels
cv2.imwrite(os.path.join(folder_name, object_name + '_HSVinBGR.jpg'), HSVinBGR)
cv2.imwrite(os.path.join(folder_name, object_name + '_intensity.jpg'), Intensity)
cv2.imwrite(os.path.join(folder_name, object_name + '_DoLP.jpg'), 255*DoLP)
cv2.imwrite(os.path.join(folder_name, object_name + '_AoP.jpg'), 255*AoP)

#perform median blur on HSVinBGR image
img = cv2.medianBlur(HSVinBGR,9)

#SIFT
gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create(100000)
kp = sift.detect(img, None)
img_keypts = cv2.drawKeypoints(img, kp, img)
edges_sift = cv2.Canny(img_keypts, 200, 250, apertureSize = 3)
cv2.imwrite('vial7_SIFTkeypts.jpg', img_keypts)

#ORB
orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
kp = orb.detect(img,None)
kp, des = orb.compute(img, kp)
orb_img = cv2.drawKeypoints(img,kp,img,color=(0,255,0), flags=0)
edges_orb = cv2.Canny(orb_img, 200, 250, apertureSize = 3)
cv2.imwrite('vial7_ORBkeypts.jpg', orb_img)

height, width = img.shape[:2]
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', width, height)
cv2.imshow('image', edges_sift)
cv2.waitKey(0)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', width, height)
cv2.imshow('image', edges_orb)
cv2.waitKey(0)

#displayImages(Intensity, AoP, DoLP, new_image, False)  #call displayimages to show each channel
# displayImages(Intensity, AoP, DoLP, new_image, True) #call displayimages to show edges of each channel
