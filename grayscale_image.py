"""Quick script: convert an image to grayscale, binarize it with a live
threshold slider (0-1, scaled to 0-255 under the hood), then apply
erosion/dilation with live sliders for kernel size and iterations."""

import cv2
import numpy as np

INPUT_PATH = r"C:\Users\ykulk\Downloads\Yash Kulkarni.png"
OUTPUT_PATH = r"C:\Users\ykulk\Downloads\Yash Kulkarni_gray.png"
BINARY_OUTPUT_PATH = r"C:\Users\ykulk\Downloads\Yash Kulkarni_binary.png"
MORPH_OUTPUT_PATH = r"C:\Users\ykulk\Downloads\Yash Kulkarni_morph.png"
BOUNDARY_OUTPUT_PATH = r"C:\Users\ykulk\Downloads\Yash Kulkarni_boundary.png"
CONV_OUTPUT_PATH = r"C:\Users\ykulk\Downloads\Yash Kulkarni_conv.png"

SLIDER_MAX = 100  # slider steps map to threshold 0.00-1.00 in increments of 0.01
MAX_KERNEL_SIZE = 21
MAX_ITERATIONS = 10
MAX_BLOCK_STEP = 20  # block size = step * 2 + 3, i.e. 3..43
MAX_C = 20  # C slider steps 0..40 map to C = step - 20, i.e. -20..20
WINDOW_NAME = "Threshold + Morphology"
CONV_WINDOW_NAME = "Convolution Kernel (setosa.io/ev/image-kernels demo)"

# 3x3 convolution kernels, same presets as https://setosa.io/ev/image-kernels/
CONV_KERNELS = [
    ("Identity", np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)),
    ("Sharpen", np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)),
    ("Box Blur", np.ones((3, 3), dtype=np.float32) / 9),
    ("Gaussian Blur", np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16),
    ("Emboss", np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)),
    ("Outline", np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)),
    ("Sobel X", np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)),
    ("Sobel Y", np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)),
]

image = cv2.imread(INPUT_PATH)
if image is None:
    raise FileNotFoundError(f"Could not read image at {INPUT_PATH}")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(OUTPUT_PATH, gray_image)
print(f"Grayscale image saved to {OUTPUT_PATH}")

filtered_gray = gray_image  # after the convolution-kernel preset
binary_image = gray_image  # after thresholding only
result_image = gray_image  # after thresholding + erode/dilate
boundary_image = gray_image  # dilate(result) - result, when boundary mode is on
display_image = gray_image  # whatever is currently shown (result or boundary)


def update(_=None):
    global filtered_gray, binary_image, result_image, boundary_image, display_image

    conv_idx = cv2.getTrackbarPos("Conv filter", WINDOW_NAME)
    conv_name, conv_kernel = CONV_KERNELS[conv_idx]
    filtered_gray = cv2.filter2D(gray_image, -1, conv_kernel)

    conv_preview = cv2.cvtColor(filtered_gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(conv_preview, conv_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow(CONV_WINDOW_NAME, conv_preview)

    adaptive_on = cv2.getTrackbarPos("Adaptive", WINDOW_NAME)
    if adaptive_on:
        block_pos = cv2.getTrackbarPos("Block size", WINDOW_NAME)
        block_size = block_pos * 2 + 3  # force odd, >= 3
        c_pos = cv2.getTrackbarPos("C", WINDOW_NAME)
        c_value = c_pos - MAX_C
        binary_image = cv2.adaptiveThreshold(
            filtered_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_value
        )
    else:
        threshold_pos = cv2.getTrackbarPos("Threshold", WINDOW_NAME)
        threshold_value = (threshold_pos / SLIDER_MAX) * 255
        _, binary_image = cv2.threshold(filtered_gray, threshold_value, 255, cv2.THRESH_BINARY)

    kernel_pos = cv2.getTrackbarPos("Kernel size", WINDOW_NAME)
    kernel_size = kernel_pos * 2 + 1  # force odd, >= 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    erode_iters = cv2.getTrackbarPos("Erode", WINDOW_NAME)
    dilate_iters = cv2.getTrackbarPos("Dilate", WINDOW_NAME)

    result_image = binary_image
    if erode_iters > 0:
        result_image = cv2.erode(result_image, kernel, iterations=erode_iters)
    if dilate_iters > 0:
        result_image = cv2.dilate(result_image, kernel, iterations=dilate_iters)

    boundary_on = cv2.getTrackbarPos("Boundary", WINDOW_NAME)
    if boundary_on:
        dilated = cv2.dilate(result_image, kernel, iterations=1)
        boundary_image = cv2.subtract(dilated, result_image)
        display_image = boundary_image
    else:
        display_image = result_image

    cv2.imshow(WINDOW_NAME, display_image)


cv2.namedWindow(WINDOW_NAME)
cv2.createTrackbar("Conv filter", WINDOW_NAME, 0, len(CONV_KERNELS) - 1, update)
cv2.createTrackbar("Threshold", WINDOW_NAME, 50, SLIDER_MAX, update)
cv2.createTrackbar("Adaptive", WINDOW_NAME, 0, 1, update)
cv2.createTrackbar("Block size", WINDOW_NAME, 4, MAX_BLOCK_STEP, update)  # default block = 11
cv2.createTrackbar("C", WINDOW_NAME, MAX_C, MAX_C * 2, update)  # default C = 0
cv2.createTrackbar("Kernel size", WINDOW_NAME, 1, MAX_KERNEL_SIZE // 2, update)
cv2.createTrackbar("Erode", WINDOW_NAME, 0, MAX_ITERATIONS, update)
cv2.createTrackbar("Dilate", WINDOW_NAME, 0, MAX_ITERATIONS, update)
cv2.createTrackbar("Boundary", WINDOW_NAME, 0, 1, update)
update()

print(
    "Drag the sliders (Conv filter 0-7, Threshold, Adaptive 0/1, Block size, C, "
    "Kernel size, Erode, Dilate, Boundary 0/1). Conv filter picks a 3x3 convolution "
    "preset (Identity/Sharpen/Box Blur/Gaussian Blur/Emboss/Outline/Sobel X/Sobel Y, "
    "same as https://setosa.io/ev/image-kernels/) applied before thresholding. "
    "When Adaptive=1, Threshold is ignored and the local Block size/C sliders are "
    "used instead. Press 's' to save, any other key to exit."
)
key = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()

if key == ord("s"):
    cv2.imwrite(CONV_OUTPUT_PATH, filtered_gray)
    cv2.imwrite(BINARY_OUTPUT_PATH, binary_image)
    cv2.imwrite(MORPH_OUTPUT_PATH, result_image)
    cv2.imwrite(BOUNDARY_OUTPUT_PATH, boundary_image)
    print(f"Convolution-filtered image saved to {CONV_OUTPUT_PATH}")
    print(f"Binary image saved to {BINARY_OUTPUT_PATH}")
    print(f"Morphology result saved to {MORPH_OUTPUT_PATH}")
    print(f"Boundary image saved to {BOUNDARY_OUTPUT_PATH}")
