# Cervical Cancer Detection Using Multimodal Medical Images

## 📌 Overview
This project aims to detect the severity of cervical cancer using deep learning models trained on multimodal medical images — specifically, **Pap smear** and **Colposcopy** images. The system classifies images into three diagnostic categories: **Normal**, **Mild**, and **Severe**, providing an intelligent decision support tool for medical professionals.

---

## 🧠 Motivation
Cervical cancer remains a significant health challenge, especially in developing countries. Early and accurate diagnosis can drastically improve survival rates. Traditional diagnostic processes are time-consuming and subject to human error. This project leverages AI to automate and enhance the diagnostic pipeline using deep learning.

---

## 🛠️ Technologies Used

- **Programming Language**: MATLAB
- **Frameworks/Toolkits**: Deep Learning Toolbox, Image Processing Toolbox
- **Model Architectures**: CNN, Vision Transformer (ViT), Attention Network
- **Dataset**: Custom structured multimodal dataset (Pap + Colposcopy)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

---

## 🗂️ Project Structure

AttentionNetwork/ # Contains MATLAB code and model files MalhariDataset/ │ ├── Pap/ │ ├── HSIL/ │ ├── LSIL/ │ ├── NILM/ │ └── SSC/ │ ├── Colposcopic/ │ ├── CN1/ │ ├── CN2/ │ └── CN3/ │ └── Main Data/ ├── Normal/ ├── Mild/ └── Severe/



---

## ⚙️ How It Works

1. **Image Acquisition**: Collects and loads paired Pap and Colposcopy images for each patient.
2. **Preprocessing**: Enhances image quality using normalization, noise reduction, etc.
3. **Feature Extraction**: Deep networks extract meaningful features.
4. **Classification**: Multiclass classifier assigns input to Normal, Mild, or Severe.
5. **Performance Evaluation**: Accuracy and other metrics are computed post-training.
6. **GUI Interface**: Allows easy image upload and result visualization.

--





