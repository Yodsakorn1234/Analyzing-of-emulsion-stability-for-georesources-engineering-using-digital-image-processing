import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# Load the image
image_path = " file Picture of referance "
image = cv2.imread(image_path)

# Convert the image to HSV for easier color detection
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the red color range in HSV
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

# Combine the masks for red color
red_mask = mask1 + mask2

# Find contours of the red areas
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Check if we found at least two contours
if len(contours) >= 2:
    # Sort the contours by area and keep the largest two
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # Calculate the closest distance between the two largest contours
    min_distance = float('inf')
    closest_points = None

    # Iterate through points on the two contours to find the minimum distance
    for pt1 in contours[0]:
        for pt2 in contours[1]:
            dist = math.dist(pt1[0], pt2[0])
            if dist < min_distance:
                min_distance = dist
                closest_points = (tuple(pt1[0]), tuple(pt2[0]))

    # Draw a green line at the midpoint of the two closest points
    if closest_points:
        mid_point = (
            (closest_points[0][0] + closest_points[1][0]) // 2,
            (closest_points[0][1] + closest_points[1][1]) // 2
        )

        # Draw the line
        cv2.line(image, closest_points[0], closest_points[1], (0, 255, 0), 2)

        # Define the known length of the needle in mm
        known_length_mm = 0.7174  # Actual length of the needle

        # Assuming you measured the needle in pixels previously (e.g., 150 pixels)
        # You'll need to set this value based on your image
        needle_length_pixels = min_distance  # Use the min_distance here

        # Calculate the conversion factor (pixels per mm)
        pixels_per_mm = needle_length_pixels / known_length_mm

        # Calculate the distance in millimeters
        distance_mm = min_distance / pixels_per_mm

        # Adjust the text position to appear beside the green line
        offset_x = 10  # Horizontal offset
        offset_y = 10  # Shift text slightly down for better alignment

        # Set the text position beside the midpoint
        text_position = (mid_point[0] + offset_x, mid_point[1] + offset_y)

        # Label the distance beside the line
        cv2.putText(image, f"{distance_mm:.2f} mm", text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the updated image
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Size: {distance_mm:.2f} mm")
        plt.axis('off')
        plt.show()
    else:
        print("Could not find closest points.")
else:
    print("Not enough contours detected.")
