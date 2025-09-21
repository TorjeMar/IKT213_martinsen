import cv2
import numpy as np

def sobel_edge_detection(image):
    blur = cv2.GaussianBlur(image,(3,3),0)
    sobel_xy = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    cv2.imwrite("images/sobel_edge_detection.png", sobel_xy)

def canny_edge_detection(image, threshold_1, threshold_2):
    blur = cv2.GaussianBlur(image,(3,3),0)
    canny = cv2.Canny(blur, threshold1=threshold_1, threshold2=threshold_2)
    cv2.imwrite("images/canny_edge_detection.png", canny)
    
def template_match(image, template):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(image_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    cv2.imwrite("images/template_match.png", image)


def resize(image, scale_factor: int, up_or_down: str):
    if up_or_down == "up":
        resized_image = image.copy()
        for _ in range(scale_factor - 1):
            resized_image = cv2.pyrUp(resized_image)
    elif up_or_down == "down":
        resized_image = image.copy()
        for _ in range(scale_factor - 1):
            resized_image = cv2.pyrDown(resized_image)
    
    cv2.imwrite("images/resized_image.png", resized_image)


def main():
    img = cv2.imread('lambo.png')
    sobel_edge_detection(img)
    canny_edge_detection(img, 50, 50)

    shape = cv2.imread("shapes-1.png")
    shape_template = cv2.imread("shapes_template.jpg", 0)
    template_match(shape, shape_template)

    resize(img, scale_factor=2, up_or_down="down")

if __name__ == '__main__':
    main()