import os
import cv2
import numpy as np

def snake_contours_A(image, contours, mm_per_pixel=0.00661):
    # แสดงข้อมูลพื้นที่ในหน่วย mm^2
    font = cv2.FONT_HERSHEY_SIMPLEX
    areas = []
    
    for i, contour in enumerate(contours, 1):
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            area_px = cv2.contourArea(contour)  # Get the area in pixels
            area_mm2 = area_px * (mm_per_pixel ** 2)  # Convert area to square millimeters
            areas.append(area_mm2)
            text = str(i)
            text1 = f"Area {i} = {area_mm2:.6f} mm^2"
            fontScale = 0.5
            thickness = 1
            cv2.putText(image, text, (cx, cy), font, fontScale, (255, 0, 0), thickness, cv2.LINE_AA)
            print(text1)  # แสดงข้อมูลผ่านคอนโซล

    # Calculate and print the average area
    if areas:
        avg_area = sum(areas) / len(areas)
        print(f"Average area: {avg_area:.6f} mm^2")
    
    print(f"Total number of shapes detected: {len(contours)}")


# Folder paths
input_folder = "snake2count/"  # โฟลเดอร์ที่มีภาพต้นฉบับ
output_folder_images = "FINAL_output/"  # โฟลเดอร์สำหรับบันทึกภาพที่ประมวลผลแล้ว

# Ensure the output folders exist
os.makedirs(output_folder_images, exist_ok=True)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(input_folder, filename)
        
        # โหลดรูปภาพ
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # เบลอและตรวจจับขอบ
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        canny = cv2.Canny(blur, 30, 150, 3)
        dilated = cv2.dilate(canny, (1, 1), iterations=0)

        # หาคอนทัวร์
        (cnt, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # วาดคอนทัวร์บนภาพ
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)

        # ใส่หมายเลขคอนทัวร์และแสดงผลข้อมูลพื้นที่
        snake_contours_A(rgb, cnt)

        # ตั้งชื่อไฟล์ output และบันทึกภาพที่ประมวลผลแล้ว
        output_image_path = os.path.join(output_folder_images, "Result_" + filename)
        cv2.imwrite(output_image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"Processed and saved image: {output_image_path}")
