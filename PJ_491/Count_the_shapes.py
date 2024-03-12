import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

def snake_contours_A(image, contours, dpi):
    # Convert dpi to mm/pixel
    if dpi is None:
        dpi = 96  # Default dpi

    mm_per_pixel = 25.4 / dpi
    
    # Draw text showing the count of shapes
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, contour in enumerate(contours, 1):
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            area_px = cv2.contourArea(contour)
            area_mm = area_px * mm_per_pixel ** 2  # Convert area to mm^2
            text = str(i)
            text1 = f"{i}: Area = {area_mm:.2f} mm^2"
            fontScale = 0.5
            thickness = 1
            cv2.putText(image, text, (cx, cy), font, fontScale, (255, 0, 0), thickness, cv2.LINE_AA)
            print(text1)  # Print area value

    # Print the total number of contours detected
    print(f"Total number of shapes detected: {len(contours)}")

# Load image
image = cv2.imread('image.png') 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

# Blur and edge detection
blur = cv2.GaussianBlur(gray, (11, 11), 0) 
canny = cv2.Canny(blur, 30, 150, 3) 
dilated = cv2.dilate(canny, (1, 1), iterations=0) 

# Find contours
(cnt, hierarchy) = cv2.findContours( 
    dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

# Draw contours on image
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2) 

# Annotate contours with numbers
snake_contours_A(rgb, cnt, dpi=96)  # Assuming 96 dpi

# Display the image with contours and count

cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close all OpenCV windows
plt.figure("Shape detection")
plt.xticks([])
plt.yticks([])
plt.imshow(rgb)
plt.title('Final')
plt.show()

