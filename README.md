# 🌱 Plant Disease Detection Using CNN

## 📌 Project Overview

Plant Disease Detection Using CNN is a deep learning-based web application that identifies plant diseases from leaf images. The system uses a Convolutional Neural Network (CNN) trained on the PlantVillage dataset to classify plant leaves into different disease categories and provide detailed disease information.

The application allows users to upload an image or capture a photo using a camera, making disease detection simple and accessible.

---

## 🎯 Features

✅ Upload leaf images for disease detection

✅ Capture leaf images using camera

✅ Disease prediction using a trained CNN model

✅ Confidence score visualization

✅ Disease probability distribution chart

✅ Scientific name of the disease

✅ Pathogen type identification

✅ Symptoms description

✅ Disease management recommendations

✅ Prevention methods

✅ Downloadable disease report

✅ Model performance visualization

---

## 📂 Dataset

**Dataset Used:** PlantVillage Dataset

### Classes Used

* Potato Early Blight
* Potato Healthy
* Tomato Early Blight
* Tomato Late Blight
* Tomato Healthy

Total Images Used: **5652+**

---

## 🧠 Model Architecture

The model is built using a Convolutional Neural Network (CNN) consisting of:

* Convolution Layers
* Max Pooling Layers
* Flatten Layer
* Dense Layers
* Softmax Output Layer

### Working Process

1. User uploads a leaf image.
2. Image is resized to 128 × 128 pixels.
3. Pixel values are normalized.
4. CNN extracts image features.
5. Softmax layer predicts the disease class.
6. Disease information is displayed.

---

## 🛠 Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Pandas
* Streamlit
* Matplotlib

---

## 📊 Model Performance

| Metric              | Value        |
| ------------------- | ------------ |
| Training Accuracy   | 95.38%       |
| Validation Accuracy | 94.69%       |
| Number of Classes   | 5            |
| Dataset             | PlantVillage |

### Evaluation Metrics

* Accuracy
* Loss Analysis
* Probability Distribution
* Confusion Matrix

---

## 📸 Application Screenshots

### Home Page

<img width="712" height="603" alt="Screenshot 2026-06-21 015620" src="https://github.com/user-attachments/assets/41d7228f-9a1a-4f8f-9645-c2be380c1b04" />


### Disease Prediction

<img width="1192" height="493" alt="Screenshot 2026-06-21 025901" src="https://github.com/user-attachments/assets/baa39774-30b7-4a0f-8ecd-5b70eaf9cc4a" />


### Probability Distribution Chart

<img width="298" height="561" alt="Screenshot 2026-06-21 025002" src="https://github.com/user-attachments/assets/e434ff41-ae56-46a5-9597-7145a64713b1" />



## 🚀 How to Run

### Clone Repository

```bash
git clone <repository-link>
cd Plant-Disease-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python -m streamlit run app.py
```

---

## 🔮 Future Enhancements

* Real-Time Live Disease Scanning
* Farmer Assistance Mode
* Student Learning Mode
* Mobile Application Development
* Multi-Language Support
* Advanced Disease Classification
* Cloud Deployment

---

## 👨‍💻 Developer

**D Haritha**

AI/ML Internship Project – 2026
