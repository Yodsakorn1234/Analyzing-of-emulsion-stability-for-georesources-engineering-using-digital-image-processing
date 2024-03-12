import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("image.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
canny = cv2.Canny(img, 100, 200)

# Create a subplot for the Canny edge-detected image
plt.imshow(canny, 'gray')
plt.title('Canny')
plt.xticks([]), plt.yticks([])

# Show only the Canny edge-detected image in a new window
plt.show()

