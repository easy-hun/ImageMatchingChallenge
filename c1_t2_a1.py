import os
import cv2
import numpy as np
import joblib
import csv
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

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
    if image is None:
        raise ValueError(f"Error: Unable to read image at path {image_path}")
    equalized_image = histogram_equalization_color(image)
    laws_features = laws_texture_energy(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2GRAY))
    glcm_feats = glcm_features(equalized_image)
    combined_features = np.concatenate((laws_features, glcm_feats))
    normalized_features = normalize_features(combined_features)
    return normalized_features

# Load the classifier
model_path = 'classifier_model.joblib'
classes_path = 'classes.joblib'
classifier = joblib.load(model_path)
classes = joblib.load(classes_path)

# Function to classify a new image with probabilities
def classify_image_with_probabilities(image_path, classifier):
    feature = extract_features(image_path)
    decision_function = classifier.decision_function([feature])[0]
    probabilities = decision_function.argsort()[-10:][::-1]
    return probabilities

# Define the path to save the CSV file
csv_file_path = 'c1_t2_a1.csv'

# Classify new images and save predictions
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    for i in range(1, 101):
        new_image_path = f'./query/query{i:03d}.png'
        new_image_name = f'query{i:03d}.png'
        if not os.path.exists(new_image_path):
            print(f"Warning: {new_image_path} does not exist")
            continue

        try:
            probabilities = classify_image_with_probabilities(new_image_path, classifier)
            top_classes = [classes[p] for p in probabilities]
            writer.writerow([new_image_name] + top_classes)
        except ValueError as e:
            print(e)

print(f"Top 10 predictions saved to {csv_file_path}")