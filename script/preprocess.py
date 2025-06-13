import os
import cv2
import numpy as np

def load_images(data_dir, img_size=100):
    X, y = [], []
    class_names = os.listdir(data_dir)
    for label, class_name in enumerate(class_names):
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_folder): continue
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(label)
            except:
                print(f"Error loading {img_path}")
    return np.array(X), np.array(y), class_names
