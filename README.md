# 🦊 ShadowFox – Image Classification using CNN

ShadowFox is a deep learning project built using **TensorFlow** and **Streamlit**, designed to classify images from the **CIFAR-10 dataset** into one of 10 categories such as airplanes, cars, birds, cats, and more.  

This project demonstrates training a **Convolutional Neural Network (CNN)** and deploying it as a **web app** using Streamlit.

You can access the live demo here:
 👉 [ShadowFox Web App](https://shadowfox-image-classifier.streamlit.app/) 

## 🚀 Features

- CNN model trained on the CIFAR-10 dataset  
- Achieves ~75% accuracy on validation data  
- Interactive Streamlit web app for real-time image prediction  
- Clean and minimal UI  
- Deployed via **Streamlit Cloud**

---

## 🧠 Dataset

We use the **CIFAR-10** dataset, which contains:
- **60,000** 32x32 color images  
- **10 classes** with **6,000 images per class**  
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck  

---

## 🧩 Model Architecture

A simple CNN built using TensorFlow/Keras:
- Convolutional layers with ReLU activation  
- MaxPooling layers  
- Dropout for regularization  
- Fully connected Dense layers  
- Softmax output layer for classification  

Trained using **Adam optimizer** and **categorical cross-entropy loss**.

---

## 📁 Project Structure

ShadowFox/

│

├── app.py # Streamlit app file

├── model.h5 # Trained CNN model

├── requirements.txt # Dependencies for deployment

├── cifar10_classify.ipynb # Jupyter Notebook for training

├── README.md # Project documentation

└── sample_images/ # Example images (optional)

---

## ⚙️ Installation & Setup (Run Locally)


Clone the repository:
```bash
git clone https://github.com/ShreySharma15/ShadowFox.git
cd ShadowFox
```
   
Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## 🌐 Deployment
The project is deployed using Streamlit Cloud.

---

## 📦 Requirements
- nginx
- streamlit
- tensorflow
- numpy
- matplotlib
- scikit-learn

(These are listed in requirements.txt)

---

## 📊 Results
- Metric	Accuracy
- Training Accuracy	~72%
- Validation Accuracy	~75%
- Test Accuracy	~73%

---

## 👨‍💻 Author
Shrey Sharma


📍 SRM University, KTR Campus

🧠 AI/ML Enthusiast | Developer | Innovator


GitHub: @ShreySharma15

---

## 🧾 License
This project is licensed under the MIT License – feel free to use and modify it for your own work.

⭐ If you like this project, give it a star on GitHub! ⭐
