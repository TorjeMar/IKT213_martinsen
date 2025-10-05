import cv2
import numpy as np

def padding(image, border_with):
    padded_image = cv2.copyMakeBorder(image, border_with, border_with, border_with, border_with, cv2.BORDER_REFLECT_101)
    cv2.imwrite("images/padded_image.png", padded_image)

def crop(image, x_0, x_1,  y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    cv2.imwrite("images/cropped_image.png", cropped_image)

def resize(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    cv2.imwrite("images/resized_image.png", resized_image)

def copy(image, emptyPictureArray):
    height, width = image.shape[:2]
    for y in range(height):
        for x in range(width):
            emptyPictureArray[y, x] = image[y, x]
    cv2.imwrite("images/copied_image.png", emptyPictureArray)

def grayscale(image):
    grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("images/grayscaled_image.png", grayscaled_image)

def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("images/hsv_image.png", hsv_image)

def hue_shifted(image, emptyPictureArray, hue):
    shifted = image.astype(np.int16) + int(hue)
    shifted = np.clip(shifted, 0, 255).astype(np.uint8)
    emptyPictureArray[:] = shifted
    cv2.imwrite("images/hue_shifted_image.png", emptyPictureArray)

def smoothing(image):
    smoothed_image = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)
    cv2.imwrite("images/smoothed_image.png", smoothed_image)

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    cv2.imwrite("images/rotated_image.png", rotated_image)

if __name__ == "__main__":
    image = cv2.imread("lena-1.png")
    height, width = image.shape[:2]
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    padding(image, 100)
    crop(image, 80, 382, 80, 382)
    resize(image, 200, 200)
    copy(image, emptyPictureArray)
    grayscale(image)
    hsv(image)
    hue_shifted(image, emptyPictureArray, 50)
    smoothing(image)
    rotation(image, 180)