import cv2
import numpy as np
import matplotlib.pyplot as plt

#function to display the different image types
def displayImages(Intensity, AoP, DoLP, HSVinBGR):
    images = [Intensity, AoP, DoLP, HSVinBGR]
    plot_titles = ['1-channel: Intensity', '1-channel: AoP', '1-channel: DoLP', 'HSV Image in BGR']
    rows = 2
    cols = 2
    plt.figure("Polarization Parameters")
    for i in range(rows*cols):
        img = images[i]
        subplot_title = plot_titles[i]
        plt.subplot(2,2,i+1)
        plt.imshow(img)
        plt.title(subplot_title)
    plt.show()

#replace file names here with the three images you want in your folder
file_image0 = 'spoon_0.jpg'
file_image45 = 'spoon_45.jpg'
file_image90 = 'spoon_90.jpg'

#reading images in bgr format and converting to grayscale
image0 = np.int16(cv2.cvtColor(cv2.imread(file_image0),cv2.COLOR_BGR2GRAY))
image45 = np.int16(cv2.cvtColor(cv2.imread(file_image45),cv2.COLOR_BGR2GRAY))
image90 = np.int16(cv2.cvtColor(cv2.imread(file_image90),cv2.COLOR_BGR2GRAY))

#calculating stokes parameters
I = image0 + image90
Q = image0 - image90
U = 2*image45 - image0 - image90
Intensity = np.sqrt(np.square(Q) + np.square(U))
DoLP = Intensity/I
AoP = 0.5*np.arctan2(U,Q)

#Use polarization info to convert to HSV
H = np.uint8(AoP*(180/np.pi))
S = np.uint8(255*(DoLP/np.amax(DoLP)))
V = np.uint8(255*(Intensity/np.amax(Intensity)))
HSV = cv2.merge((H,S,V))
new_image = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)

#Extract individual channels
cv2.imwrite('intensity.jpg', Intensity)
cv2.imwrite('DoLP.jpg', 255*DoLP)
cv2.imwrite('AoP.jpg', 255*AoP)

height, width = new_image.shape[:2]
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', width, height)
cv2.imshow('image', new_image)
cv2.waitKey(0)

displayImages(Intensity, AoP, DoLP, new_image) #call the displayImages function
