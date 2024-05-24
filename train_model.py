import csv
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
def load_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(row[0])
            data.append([float(x) for x in row[1:]])

    return np.array(data), np.array(labels)

data, labels = load_data('data/sign_data.csv')

# Convert labels to one-hot encoding
unique_labels = sorted(set(labels))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
labels = np.array([label_map[label] for label in labels])
labels = to_categorical(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(unique_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save('sign_language_model.h5')

print("Model trained and saved as sign_language_model.h5")
