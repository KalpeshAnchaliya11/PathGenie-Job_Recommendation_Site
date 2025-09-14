import os

import joblib
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Create directory to save models if not exists
os.makedirs("models", exist_ok=True)

# Load and prepare dataset
df = pd.read_csv("resume_dataset.csv")
texts = df['resume_text']

labels = df['role']
# Encode target labels
label_encoder = LabelEncoder()

labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)


# Vectorize resume texts using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

features = vectorizer.fit_transform(texts).toarray()
# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(

    features, labels_categorical, test_size=0.2, random_state=42


)
# Define a simple neural network model
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),

    Dropout(0.4),

    Dense(256, activation='relu'),


    Dropout(0.3),
    Dense(labels_categorical.shape[1], activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Train the model
history = model.fit(
    X_train, y_train,

    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    verbose=1
)




# Save trained components

model.save("models/nn_model.h5")


joblib.dump(vectorizer, "models/nn_vectorizer.pkl")
joblib.dump(label_encoder, "models/nn_label_encoder.pkl")
# Plot training history
plt.figure(figsize=(8, 5))

plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend()


plt.grid(True)
plt.tight_layout()

plt.savefig("nn_accuracy_plot.png")
plt.close()
print("model training complete.files saved in 'models/' directory.")

print("accuracy plot saved as'nn_accuracy_plot.png'")
