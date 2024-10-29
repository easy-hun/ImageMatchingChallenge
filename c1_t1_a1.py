# c1_t1_a1.py
# [Feature]
# -> Color 정규화 + Law's Texture + GLCM

import os
import cv2
import numpy as np
import joblib
import csv
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import MinMaxScaler
from scipy import signal as sg
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

### Color Normalization ###
def histogram_equalization_color(image):
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_img)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq_img = cv2.merge((y_eq, cr, cb))
    bgr_eq_img = cv2.cvtColor(ycrcb_eq_img, cv2.COLOR_YCrCb2BGR)
    
    return bgr_eq_img

### Feature Normalization ###
def normalize_features(features):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized = scaler.fit_transform(features.reshape(-1, 1)).flatten()
    return normalized

### Law's Texture ###
def laws_texture_energy(image):
    kernels = {
        'L5': np.array([1, 4, 6, 4, 1]),
        'E5': np.array([-1, -2, 0, 2, 1]),
        'S5': np.array([-1, 0, 2, 0, -1]),
        'W5': np.array([-1, 2, 0, -2, 1]),
        'R5': np.array([1, -4, 6, -4, 1])
    }
    filters = []
    for k1 in kernels:
        for k2 in kernels:
            filters.append(np.outer(kernels[k1], kernels[k2]))

    energy_maps = []
    for kernel in filters:
        filtered = cv2.filter2D(image, -1, kernel)
        energy = np.abs(filtered)
        energy_maps.append(energy)

    energy_features = [np.mean(em) for em in energy_maps]
    return np.array(energy_features)

### GLCM ###
def glcm_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    asm = graycoprops(glcm, 'ASM').mean()
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation, asm])

### 특징 결합 ###
def extract_features(image_path):
    image = cv2.imread(image_path)
    equalized_image = histogram_equalization_color(image)
    laws_features = laws_texture_energy(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2GRAY))
    glcm_feats = glcm_features(equalized_image)
    combined_features = np.concatenate((laws_features, glcm_feats))
    normalized_features = normalize_features(combined_features)
    return normalized_features


'''
### 학습 ~ ###
def load_dataset(dataset_path):
    classes = os.listdir(dataset_path)
    features = []
    labels = []
    for label, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                feature = extract_features(image_path)
                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels), classes

dataset_path = './TrainImages'
features, labels, classes = load_dataset(dataset_path)

# Check class distribution
print("Class distribution:", Counter(labels))

# Train a classifier
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Test the classifier
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the classifier
model_path = 'classifier_model.joblib'
classes_path = 'classes.joblib'
joblib.dump(classifier, model_path)
joblib.dump(classes, classes_path)
print(f"Model saved to {model_path} and classes saved to {classes_path}")

# Load the classifier
model_path = 'classifier_model.joblib'
classes_path = 'classes.joblib'
classifier = joblib.load(model_path)
classes = joblib.load(classes_path)
print(f"Model loaded from {model_path}")

# Evaluate the classifier
y_pred_train = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)

# Function to classify a new image
def classify_image(image_path, classifier):
    feature = extract_features(image_path)
    prediction = classifier.predict([feature])
    return prediction
'''


# 챌린지 코드 ~
### New Image Classify ###
model_path = 'classifier_model.joblib'
classes_path = 'classes.joblib'
classifier = joblib.load(model_path)
classes = joblib.load(classes_path)

# List to store predictions
predictions = []

# Function to classify a new image
def classify_image(image_path, classifier):
    feature = extract_features(image_path)
    prediction = classifier.predict([feature])
    return prediction

# Classify a new image and save predictions
for i in range(1, 101): # 챌린지 시에 101로 수정
    new_image_path = f'./query/query{i:03d}.png'
    #new_image_path = f'./query/query ({i}).png'
    new_image_name = f'query{i:03d}.png'
    prediction = classify_image(new_image_path, classifier)
    predicted_class = classes[prediction[0]]
    predictions.append({'Image': new_image_name, 'Predicted_Class': predicted_class})

# Define the path to save the CSV file
csv_file_path = 'c1_t1_a1.csv'

# Write predictions to CSV file without headers
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write predictions to CSV file
    for prediction in predictions:
        writer.writerow([prediction['Image'], prediction['Predicted_Class']])

print(f"Predictions saved to {csv_file_path}")
