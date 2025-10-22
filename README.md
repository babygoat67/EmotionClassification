Emotion Detection using CNN (TensorFlow)

This project uses a **Convolutional Neural Network (CNN)** built with **TensorFlow and Keras** to classify human emotions (happy 😄 or sad 😢) from images.  

---

## 📁 Project Structure
```
emotion-detection/
│
├── data/
│   ├── happy/
│   └── sad/
│
├── models/
│   └── emotionmodel.h5
│
├── happytest2.jpeg
├── imageClassification.ipynb
└── README.md
```

---

## 🧠 Overview
The model is trained on a custom dataset containing **happy** and **sad** facial expressions.  
It uses **OpenCV** for image handling and **TensorFlow** for model training.  
After training, the model predicts whether a new image shows a happy or sad emotion.

---

## ⚙️ Setup Instructions

### 1️⃣ Install Dependencies
Make sure Python 3.8+ is installed, then install all required packages:
```bash
pip install tensorflow opencv-python matplotlib
```

### 2️⃣ Dataset Preparation
Your dataset folder (data/) should contain two subfolders:
```
data/
├── happy/
│   ├── happy1.jpg
│   ├── happy2.jpg
│   └── ...
└── sad/
    ├── sad1.jpg
    ├── sad2.jpg
    └── ...
```
Each subfolder represents one class label.

### 3️⃣ Data Cleaning
The script automatically checks and removes invalid images or files not matching the allowed extensions:
```python
image_exts = ['jpeg', 'jpg', 'bmp', 'png']
```

### 4️⃣ Data Loading
Images are loaded using:
```python
data = tf.keras.utils.image_dataset_from_directory('data')
```
Images are resized to 256×256  
Labels are automatically generated (0 for happy, 1 for sad)

---

## 🧩 Model Architecture
A simple 3-layer CNN is used:

| Layer Type          | Output Shape      | Parameters |
|---------------------|-------------------|------------|
| Conv2D (16 filters) | (254, 254, 16)   | 448        |
| MaxPooling2D        | (127, 127, 16)   | 0          |
| Conv2D (32 filters) | (125, 125, 32)   | 4,640      |
| MaxPooling2D        | (62, 62, 32)     | 0          |
| Conv2D (16 filters) | (60, 60, 16)     | 4,624      |
| MaxPooling2D        | (30, 30, 16)     | 0          |
| Flatten             | (14400)          | 0          |
| Dense (256 units)   | (256)            | 3,686,656  |
| Dense (1 unit)      | (1)              | 257        |

Total parameters: 3,696,625

---

## 🧠 Model Training
The model is trained for 20 epochs with TensorBoard logging:
```python
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
```
Training and validation accuracy are plotted with Matplotlib.
<img width="616" height="521" alt="image" src="https://github.com/user-attachments/assets/9c6d879b-e8df-46d8-8d15-52efffe4f5bc" />
<img width="615" height="515" alt="image" src="https://github.com/user-attachments/assets/efc49b89-8590-47e8-b7e9-47192d825bf7" />



---

## 📊 Model Evaluation
Model performance is measured using:

- Precision
- Recall
- Binary Accuracy

Example output:
```python
Precision : 1.0, Recall : 1.0, Accuracy : 1.0
```

---

## 🧪 Testing on New Images
Test with an unseen image:
```python
img = cv.imread('happytest2.jpeg')
resized = tf.image.resize(img, (256,256))
yhat = model.predict(np.expand_dims(resized/255, 0))
```
Output:
```
happy
```
If `yhat > 0.5`, the model predicts sad, otherwise happy.

---

## 💾 Saving & Loading the Model
```python
model.save('models/emotionmodel.h5')
new_model = tf.keras.models.load_model('models/emotionmodel.h5')
```
⚠️ Note: HDF5 (.h5) format is legacy.  
You can also save using the newer .keras format:
```python
model.save('models/emotionmodel.keras')
```

---

## 📈 Example Results

| Image   | Prediction |
|---------|------------|
| Happy 😄 | Happy      |
| Sad 😢   | Sad        |

---

## 🧩 Requirements

- Python ≥ 3.8
- TensorFlow ≥ 2.10
- OpenCV ≥ 4.5
- Matplotlib ≥ 3.5
- NumPy ≥ 1.21

---

## 🚀 Future Improvements
- Add more emotion categories (e.g., angry, surprised, neutral)  
- Improve accuracy with data augmentation  
- Use pre-trained CNNs (e.g., MobileNetV2, ResNet50)  
- Deploy as a web app using Flask or Streamlit  

---

## 👨‍💻 Author
Choo Yit Shern  
📍 Universiti Sains Malaysia  
🧠 Passionate about AI, deep learning, and computer vision.
