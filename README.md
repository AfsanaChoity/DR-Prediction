#Diabetic Retinopathy Detection from OCT Images

This project focuses on detecting Diabetic Retinopathy (DR) from Optical Coherence Tomography (OCT) images using a Convolutional Neural Network (CNN). The dataset contains images of DME (Diabetic Macular Edema) and Normal retinal scans.
Workflow

Data Preprocessing

Crop retina region from OCT images.

Resize cropped images to 224x224.

Normalize pixel values to [0,1].

Data Splitting

Train: 90%

Validation: 5%

Test: 5%

Model Architecture

CNN with 3 convolutional blocks.

Fully connected layers with L2 regularization and dropout.

Output: Binary classification (DME / Normal).

Training

Loss: Binary Crossentropy

Optimizer: Adam

Epochs: 20

Callbacks: ReduceLROnPlateau for learning rate adjustment

Evaluation

Accuracy achieved: ~99% on validation set

K-fold cross-validation supported

Dependencies

Python 3.x

TensorFlow / Keras

OpenCV

NumPy, Pandas, Matplotlib, Seaborn

scikit-learn, imutils, tqdm

Usage

Place the dataset in the Dataset/ folder.

Run the preprocessing script to crop and resize images.

Train the CNN model using train_ds and validate on validation_ds.

Evaluate performance on test_ds.
