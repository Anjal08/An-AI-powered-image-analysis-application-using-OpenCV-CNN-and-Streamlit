
# 🖼️ Image Analyzer – AI-Powered Image Processing & Classification

An interactive image analysis application built using **Python, OpenCV, TensorFlow/Keras, NumPy, and Streamlit**. This project combines traditional image processing techniques with a custom-trained CNN model for image classification.

## 🚀 Features

### 🖼️ Image Processing
- Image to Grayscale Conversion
- Gaussian Blur
- Image Thresholding
- Adjustable Kernel Size

### 🔍 Edge Detection & Contour Analysis
- Canny Edge Detection
- Contour Detection
- Contour Drawing
- Bounding Box Detection
- Minimum Contour Area Filtering

### 📊 Image Statistics
- Average Brightness Calculation
- Contrast Measurement
- Grayscale Histogram Visualization

### 🤖 CNN-Based Image Classification
- Cat and Dog Image Classification
- Custom CNN Model using TensorFlow/Keras
- Prediction Confidence Score
- 128 × 128 Image Input Size

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core Programming |
| OpenCV | Image Processing |
| TensorFlow / Keras | Deep Learning and CNN |
| NumPy | Numerical Operations |
| Pillow | Image Handling |
| Streamlit | Web Application |

## 🧠 Model Information

- Model: Custom Convolutional Neural Network (CNN)
- Input Shape: 128 × 128 × 3
- Convolutional Layers: 3
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Epochs: 10
- Validation Accuracy: 72.6%

**Note:** The trained model is designed specifically for Cat and Dog classification, not general-purpose object recognition.

## 📂 Project Structure

```text
Image-Analyzer/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── models/
    └── cats_dogs_classifier.keras
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Image-Analyzer.git
```

### 2. Navigate to Project Directory

```bash
cd Image-Analyzer
```

### 3. Create Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run Application

```bash
streamlit run app.py
```

## 📸 Application Preview

Add your application screenshots here:

```markdown
![Image Analyzer Preview](screenshots/app-preview.png)
```

## 🔮 Future Improvements

- Real-Time Webcam Analysis
- Object Detection using YOLO
- More Image Classification Categories
- CNN Model Optimization
- Cloud Deployment

## 👩‍💻 Author

**Anjali Patel**

B.Tech – Computer Science & Engineering

---

⭐ If you find this project useful, consider giving it a star!
