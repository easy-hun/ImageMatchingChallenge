import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore

# Step 1: Load and preprocess image data
image_dir = "./TrainImages"
class_names = os.listdir(image_dir)
image_data = []
labels = []

for class_name in class_names:
    class_dir = os.path.join(image_dir, class_name)
    for filename in os.listdir(class_dir):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(class_dir, filename))
            img = cv2.resize(img, (120, 120))  # Resize image if necessary
            image_data.append(img)
            labels.append(class_names.index(class_name))

image_data = np.array(image_data)
labels = np.array(labels)

# Step 2: Split dataset into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(image_data, labels, test_size=0.2, random_state=42)

# Step 3: Define CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(120, 120, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Step 5: Load and predict query images
query_dir = "./query"
query_images = []
query_filenames = []

for filename in os.listdir(query_dir):
    if filename.endswith(".png"):
        img = cv2.imread(os.path.join(query_dir, filename))
        img = cv2.resize(img, (120, 120))  # Resize image if necessary
        query_images.append(img)
        query_filenames.append(filename)

query_images = np.array(query_images)
predictions = model.predict(query_images)

# Step 6: Extract top 10 predictions for each query image
top_10_predictions = [np.argsort(pred)[-10:][::-1] for pred in predictions]

# Step 7: Create a DataFrame for top 10 predictions
data = []
for i, filename in enumerate(query_filenames):
    row = [filename]
    for idx in top_10_predictions[i]:
        row.append(class_names[idx])
    data.append(row)

# Sort the data based on the numerical value in the filenames
#data.sort(key=lambda x: int(x[0].split('(')[1].split(')')[0]))
data.sort(key=lambda x: int(x[0].split('.')[0][-3:]))

# Define column names
columns = ['Image'] + [f'Top_{i+1}' for i in range(10)]

# Create DataFrame and save to CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv('c2_t2_a1.csv', index=False, header=False)

print(f"Top 10 predictions saved to 'c2_t2_a1.csv' without header")