import numpy as np
import cv2

def print_image_information(image):
    print("Height:", image.shape[0])
    print("Width:", image.shape[1])
    print("Channels:", image.shape[2] if len(image.shape) == 3 else 1)
    print("Size:", image.size)
    print("Data type:", image.dtype)

def get_camera_output():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    ret, frame = cap.read()
    height = frame.shape[0] if ret else 0
    width = frame.shape[1] if ret else 0
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
    cap.release()
    if not ret:
        print("Error: Could not read frame from camera.")
        return None
    return fps, height, width

def main():
    img = cv2.imread('lena-1.png')
    video_fps, video_height, video_width = get_camera_output()
    with open("solutions/camera_outputs.txt", "w") as f:
        f.write("fps: {}\n".format(video_fps if video_fps is not None else 0))
        f.write("height: {}\n".format(video_height if video_height is not None else 0))
        f.write("width: {}\n".format(video_width if video_width is not None else 0))
    print_image_information(img)


if __name__ == "__main__":
    main()