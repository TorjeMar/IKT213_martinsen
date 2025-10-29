import cv2
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import psutil
import gc

def preprocess_fingerprint(image_path):
    img = cv2.imread(image_path, 0)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_bin

def match_fingerprints(img1_path, img2_path, method="orb"):
    img1 = preprocess_fingerprint(img1_path)
    img2 = preprocess_fingerprint(img2_path)

    if method == "orb":
        detector = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)
        if des1 is None or des2 is None:
            return 0, None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
    else:  # sift
        detector = cv2.SIFT_create(nfeatures=1000)
        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)
        if des1 is None or des2 is None:
            return 0, None
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good_matches), match_img

def process_dataset(dataset_path, results_folder, method="orb", threshold=20):
    y_true = []
    y_pred = []
    os.makedirs(results_folder, exist_ok=True)

    print(f"Processing dataset in: {dataset_path}")

    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.tif', '.png', '.jpg'))]
            if len(image_files) != 2:
                print(f"Skipping {folder}, expected 2 images but found {len(image_files)}")
                continue

            img1_path = os.path.join(folder_path, image_files[0])
            img2_path = os.path.join(folder_path, image_files[1])
            match_count, match_img = match_fingerprints(img1_path, img2_path, method=method)

            actual_match = 1 if "same" in folder.lower() else 0
            y_true.append(actual_match)

            predicted_match = 1 if match_count > threshold else 0
            y_pred.append(predicted_match)

            result = f"{method}_matched" if predicted_match == 1 else f"{method}_unmatched"
            print(f"{folder}: {result.upper()} ({match_count} good matches)")

            if match_img is not None:
                match_img_filename = f"{folder}_{result}.png"
                match_img_path = os.path.join(results_folder, match_img_filename)
                cv2.imwrite(match_img_path, match_img)
                print(f"Saved match image at: {match_img_path}")
    
    print(f"Total pairs processed: {len(y_true)}")
    print("Accuracy:", sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true) if y_true else 0)
    print("F1 Score:", f1_score(y_true, y_pred) if y_true else 0)
    print("Recall:", sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1) / sum(1 for true in y_true if true == 1) if any(true == 1 for true in y_true) else 0)
    print("Precision:", sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1) / sum(1 for pred in y_pred if pred == 1) if any(pred == 1 for pred in y_pred) else 0)

    if len(y_true) > 0:
        labels = ["Different (0)", "Same (1)"]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        plt.figure(figsize=(6, 5))
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix - {method.upper()}")
        plt.savefig(os.path.join(results_folder, f"confusion_matrix_{method}.png"))
        plt.close()
    else:
        print(f"No valid predictions to create confusion matrix for {method}")

def run_fingerprint_analysis(dataset_path, results_folder, method="orb", threshold=20):
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    # Start CPU monitoring
    process.cpu_percent()  # Call once to initialize
    
    start_time = time.time()
    process_dataset(dataset_path, results_folder, method=method, threshold=threshold)
    end_time = time.time()
    
    # Get final memory usage and CPU usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = final_memory - initial_memory
    cpu_percent = process.cpu_percent()  # Get CPU usage since last call
    
    print(f"Time taken for {method.upper()} matching: {end_time - start_time:.2f} seconds")
    print(f"Memory used: {memory_used:.2f} MB")
    print(f"CPU usage: {cpu_percent:.2f}%")
    
    # Force garbage collection
    gc.collect()

# Run analyses
run_fingerprint_analysis("data_check", "results", "orb", 14)
run_fingerprint_analysis("data_check", "results2", "sift", 14)
run_fingerprint_analysis("uia", "results_uia", "orb", 14)
run_fingerprint_analysis("uia", "results2_uia", "sift", 14)
