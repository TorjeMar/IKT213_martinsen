import cv2
import numpy as np


def harris_corners(reference_image):
    reference_image = reference_image.copy()
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    threshold = 0.01 * dst.max()
    reference_image[dst > threshold] = [0, 0, 255]

    cv2.imwrite('harris.png', reference_image)


def align_images(image_to_align, reference_image, max_features=10, good_match_percent=0.7):
    gray1 = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < good_match_percent * n.distance:
            good_matches.append(m)

    print(f"Good matches found: {len(good_matches)}")

    if len(good_matches) < max_features:
        print(f"Not enough matches: {len(good_matches)} found, need at least {max_features}")
        return

    good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_features]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    h, w = reference_image.shape[:2]
    aligned = cv2.warpPerspective(image_to_align, M, (w, h))
    cv2.imwrite("aligned.png", aligned)

    matches_img = cv2.drawMatches(image_to_align, kp1, reference_image, kp2, good_matches, None,
                                  matchColor=(0, 255, 0), flags=2)
    cv2.imwrite("matches.png", matches_img)


def main():
    reference = cv2.imread("reference_img.png")
    image = cv2.imread("align_this.jpg")

    harris_corners(reference)
    align_images(image, reference, max_features=10, good_match_percent=0.7)


if __name__ == "__main__":
    main()
