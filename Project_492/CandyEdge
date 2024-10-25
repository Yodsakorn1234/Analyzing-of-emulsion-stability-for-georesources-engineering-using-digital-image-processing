import os
import imageio
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

# Define the Sobel filter function
def SobelFilter(img, direction):
    if direction == 'x':
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        return ndimage.convolve(img, Gx)
    elif direction == 'y':
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        return ndimage.convolve(img, Gy)
    else:
        raise ValueError("Invalid direction. Use 'x' or 'y'.")

# Define the Normalize function
def Normalize(img):
    img_min = np.min(img)
    img_max = np.max(img)
    img_normalized = (img - img_min) / (img_max - img_min)
    return img_normalized

# Non-Maximum Suppression function without interpolation
def NonMaxSupWithoutInterpol(Gmag, Grad):
    NMS = np.zeros(Gmag.shape)
    for i in range(1, int(Gmag.shape[0]) - 1):
        for j in range(1, int(Gmag.shape[1]) - 1):
            if((Grad[i,j] >= -22.5 and Grad[i,j] <= 22.5) or (Grad[i,j] <= -157.5 and Grad[i,j] >= 157.5)):
                if((Gmag[i,j] > Gmag[i,j+1]) and (Gmag[i,j] > Gmag[i,j-1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 22.5 and Grad[i,j] <= 67.5) or (Grad[i,j] <= -112.5 and Grad[i,j] >= -157.5)):
                if((Gmag[i,j] > Gmag[i+1,j+1]) and (Gmag[i,j] > Gmag[i-1,j-1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 67.5 and Grad[i,j] <= 112.5) or (Grad[i,j] <= -67.5 and Grad[i,j] >= -112.5)):
                if((Gmag[i,j] > Gmag[i+1,j]) and (Gmag[i,j] > Gmag[i-1,j])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 112.5 and Grad[i,j] <= 157.5) or (Grad[i,j] <= -22.5 and Grad[i,j] >= -67.5)):
                if((Gmag[i,j] > Gmag[i+1,j-1]) and (Gmag[i,j] > Gmag[i-1,j+1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0

    return NMS

# Double threshold Hysterisis function
def DoThreshHyst(img):
    highThresholdRatio = 0.32
    lowThresholdRatio = 0.30
    GSup = np.copy(img)
    h = int(GSup.shape[0])
    w = int(GSup.shape[1])
    highThreshold = np.max(GSup) * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio    
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if(GSup[i, j] > highThreshold):
                GSup[i, j] = 1
            elif(GSup[i, j] < lowThreshold):
                GSup[i, j] = 0
            else:
                if((GSup[i-1, j-1] > highThreshold) or 
                   (GSup[i-1, j] > highThreshold) or
                   (GSup[i-1, j+1] > highThreshold) or
                   (GSup[i, j-1] > highThreshold) or
                   (GSup[i, j+1] > highThreshold) or
                   (GSup[i+1, j-1] > highThreshold) or
                   (GSup[i+1, j] > highThreshold) or
                   (GSup[i+1, j+1] > highThreshold)):
                    GSup[i, j] = 1
    
    GSup = (GSup == 1) * GSup # This is done to remove/clean all the weak edges which are not connected to strong edges
    
    return GSup

# Folder containing images
folder_path = "Lab/"

# Process each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        img_path = os.path.join(folder_path, filename)
        
        # Read the image
        img = imageio.imread(img_path)

        # Convert the image to grayscale
        img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])

        # Blur the grayscale image using Gaussian filter
        img_gaussian = ndimage.gaussian_filter(img_gray, sigma=1)

        # Apply Sobel filter in x and y directions
        gx = SobelFilter(img_gaussian, 'x')
        gy = SobelFilter(img_gaussian, 'y')

        # Normalize the gradients
        gx_normalized = Normalize(gx)
        gy_normalized = Normalize(gy)

        # Compute magnitude of gradients for gx and gy
        Mag = np.hypot(gx, gy)

        Gmat = np.degrees(np.arctan2(gy, gx))

        WINMS = NonMaxSupWithoutInterpol(Mag, Gmat)
        WINMS = Normalize(WINMS)

        # Apply thresholding using hysteresis method
        Final_Image = DoThreshHyst(WINMS)

        # Save the final image
        output_path = os.path.join(folder_path, "Processed_" + filename)
        plt.imsave(output_path, Final_Image, cmap='gray')

        print(f"Processed and saved: {output_path}")
