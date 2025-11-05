# Diabetic Retinopathy Detection from OCT Images

A deep learning project for **detecting Diabetic Retinopathy (DR)** from Optical Coherence Tomography (OCT) images using a **Convolutional Neural Network (CNN)**. The model classifies retinal scans as **DME (Diabetic Macular Edema)** or **Normal**.  

---

## 📝 Features
- Automated detection of Diabetic Retinopathy from OCT images  
- Data preprocessing: retina cropping, resizing, and normalization  
- CNN-based architecture with 3 convolutional blocks and fully connected layers  
- Regularization using **L2** and **dropout**  
- Binary classification: **DME / Normal**  
- Training with **Adam optimizer** and **Binary Crossentropy loss**  
- Learning rate adjustment using **ReduceLROnPlateau**  
- Supports **K-fold cross-validation**  

---

## 📊 Workflow

**1. Data Preprocessing**  
- Crop retina region from OCT images  
- Resize images to 224x224  
- Normalize pixel values to [0,1]  

**2. Data Splitting**  
- Train: 90%  
- Validation: 5%  
- Test: 5%  

**3. Model Training**  
- Epochs: 20  
- Optimizer: Adam  
- Loss: Binary Crossentropy  
- Callbacks: ReduceLROnPlateau  

**4. Evaluation**  
- Accuracy: ~99% on validation set  

---

## 🛠️ Tech Stack / Dependencies
- **Python 3.x**  
- **TensorFlow / Keras**  
- **OpenCV**  
- **NumPy, Pandas, Matplotlib, Seaborn**  
- **scikit-learn, imutils, tqdm**  

---

## ⚡ Usage
1. Place the dataset in the `Dataset/` folder  
2. Run the preprocessing script to crop and resize images  
3. Train the CNN model on `train_ds` and validate on `validation_ds`  
4. Evaluate the model performance on `test_ds`  

---

## 🔍 Benefits
- Accurate and fast detection of diabetic retinopathy  
- Supports medical diagnostics with minimal manual intervention  
- Can be extended to other retinal diseases 
