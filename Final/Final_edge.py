import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_bubbles_with_laplace(image_path):
    # 1. โหลดภาพและแปลงเป็น Grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian Blur เพื่อลด noise และทำให้ภาพเรียบขึ้น
    blurred = cv2.GaussianBlur(gray, (11, 11), sigmaX=2)

    # 3. ใช้ Laplacian Filter เพื่อตรวจจับขอบ
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))  # แปลงเป็น uint8

    # 4. Thresholding เพื่อให้ขอบเด่นชัด
    _, binary = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)

    # 5. ใช้ Morphological Closing เพื่อเชื่อมขอบ
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 6. ค้นหา Contours ของฟอง
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7. วาด Contours ลงบนภาพต้นฉบับ
    output = img.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    # 8. แสดงผลลัพธ์
    plt.figure(figsize=(12, 6))

    # แสดงภาพจาก Laplacian Filter
    # `plt.subplot(1, 2, 1)` is creating a subplot grid with 1 row, 2 columns, and selecting the first
    # subplot for the current plot. In this case, it is dividing the plotting area into 1 row and 2
    # columns, and the subsequent plot commands will be applied to the first subplot (left side) of
    # this grid.
    # plt.subplot(1, 2, 1)
    plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian Filter Output')
    plt.axis('off')

    # แสดงภาพพร้อม Contours
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    # plt.title('Detected Bubbles with Contours')
    # plt.axis('off')

    plt.show()

    # แสดงจำนวนฟองที่ตรวจพบ
    print(f"จำนวนฟองที่ตรวจพบ: {len(contours)}")

# เรียกใช้ฟังก์ชันเพื่อตรวจจับฟอง
detect_bubbles_with_laplace("R4.png")
