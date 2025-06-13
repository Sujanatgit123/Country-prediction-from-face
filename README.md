# 🌍 Country Prediction from Face using CNN 🧠

This project is a deep learning-based image classification system that predicts a person's **country of origin** using only a facial image. It was developed as part of a capstone project by **Sujan Singh**, using TensorFlow, OpenCV, and Python.

---

## 📌 Project Highlights

- 🧠 Built using Convolutional Neural Networks (CNN)
- 🖼️ Image-based prediction (100x100 facial images)
- 🌐 Supports country classification: India, USA, Nigeria, China
- 🧪 Command-line prediction script
- 💾 Trained model (`.h5`) included
- 🧰 Clean architecture with test samples and clear codebase

---

## 🗂️ Project Structure

ml_project/

│

├── scripts/ # Model training, preprocessing, prediction

│ ├── train.py

│ ├── predict.py

│ └── preprocess.py

│

├── model/ # Trained CNN model (.h5)

│ └── country_model.h5

│

├── sample_test_image/ # Sample test image(s)

│

├── requirements.txt # Python dependencies

├── README.md # Project overview

├── .gitignore # Git exclusions (data, .venv, etc.)

└── .venv/ (ignored) # Python virtual environment

---

## 🚀 How to Run

### 🛠️ Step 1: Install dependencies

pip install -r requirements.txt

### 🧪 Step 2: Run the prediction script

python scripts/predict.py

---

## 🤖 Technologies Used

| Tool            | Version     |
|-----------------|-------------|
| 🐍 Python        | 3.9.13       |
| 🔶 TensorFlow    | 2.10.0       |
| 🎯 OpenCV        | 4.11         |
| 📊 NumPy         | Included     |
| 📈 Pandas        | Included     |
| 📉 Matplotlib    | Included     |
| 📚 scikit-learn  | Latest       |
| 🌐 Streamlit     | Optional GUI |

---

## 🔬 Model Details
Model: CNN with Conv2D, MaxPooling, Flatten, Dense, Softmax layers

Input Shape: 100x100x3 images

Dataset: ~100 facial images per country (India, USA, Nigeria, China)

Accuracy: ~99% on training data, ~50% on unseen validation data

---

## 📌 Future Enhancements
Increase dataset size & diversity

Add more countries and cultural styles

Build a Streamlit or Flask-based web app

Improve accuracy with transfer learning or better data augmentation

---

## 👨‍💻 Author
**Sujan Singh**

AI & Data Science Enthusiast

Github -  https://github.com/Sujanatgit123
