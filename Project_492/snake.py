mport numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the uploaded image
image_path = "/mnt/data/IMG_3305.jpeg"
Image = cv2.imread(image_path)  # Load the image
image_gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
img = np.array(image_gray, dtype=np.float64)

# Initialize the Level Set Function (LSF)
IniLSF = np.ones(img.shape, img.dtype)
IniLSF[30:80, 30:80] = -1
IniLSF = -IniLSF

# Optimized mathematical operations
def mat_math(input_array, operation):
    if operation == "atan":
        return np.arctan(input_array)
    elif operation == "sqrt":
        return np.sqrt(input_array)

# Chan-Vese Level Set evolution function
def CV(LSF, img, mu, nu, epsilon, step):
    Drc = (epsilon / np.pi) / (epsilon**2 + LSF**2)
    Hea = 0.5 * (1 + (2 / np.pi) * mat_math(LSF / epsilon, "atan"))

    Iy, Ix = np.gradient(LSF)  # Compute gradients
    s = mat_math(Ix**2 + Iy**2, "sqrt")

    Nx = Ix / (s + 1e-6)  # Normalize gradients
    Ny = Iy / (s + 1e-6)

    Nxx, _ = np.gradient(Nx)  # Curvature calculation
    _, Nyy = np.gradient(Ny)
    cur = Nxx + Nyy

    Length = nu * Drc * cur
    Lap = cv2.Laplacian(LSF, cv2.CV_64F)
    Penalty = mu * (Lap - cur)

    s1 = Hea * img  # Update region means
    s2 = (1 - Hea) * img
    C1 = s1.sum() / (Hea.sum() + 1e-6)
    C2 = s2.sum() / ((1 - Hea).sum() + 1e-6)

    CVterm = Drc * (-(img - C1)**2 + (img - C2)**2)
    LSF += step * (Length + Penalty + CVterm)  # Update LSF
    return LSF

# Parameters
mu = 1
nu = 0.003 * 255 * 255
num_iterations = 100
epsilon = 1
step = 0.1

# Initialize and evolve the Level Set Function
LSF = IniLSF
for _ in range(num_iterations):
    LSF = CV(LSF, img, mu, nu, epsilon, step)

# Plot the results with red contours
plt.imshow(cv2.cvtColor(Image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
plt.contour(LSF, [0], colors='r', linewidths=2)  # Red contours
plt.axis('off')  # Hide axis
plt.show()