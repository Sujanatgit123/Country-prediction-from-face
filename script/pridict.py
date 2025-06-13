import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import load_images
import os

# Load the trained model and class names.
model = load_model("model/country_model.h5")
_, _, class_names = load_images("../data/")  # Get class labels

def predict_country(image_path):
    image_path = image_path.strip().strip('"')

    if not os.path.isfile(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image. Check if it's corrupted or unsupported.")
        return

    img = cv2.resize(img, (100, 100))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    print(f"✅ Predicted Country: {class_names[class_id]} ({prediction[0][class_id] * 100:.2f}%)")

if __name__ == "__main__":
    print("🌍 Country Predictor from Face")
    print("📁 Type the full path to an image file. Type 'exit' to quit.\n")

    while True:
        path_input = input("Enter image path: ")
        if path_input.lower() == "exit":
            print("👋 Exiting.")
            break

        predict_country(path_input)
'''thanks'''
