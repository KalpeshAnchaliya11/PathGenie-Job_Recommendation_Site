import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


import joblib
import os



#xheck if models folder is available or not
os.makedirs("models", exist_ok=True)

#resume dataset loading process
df = pd.read_csv("resume_dataset.csv")

# 🔍 Features and Labels
X = df['resume_text']
y = df['role']

#encodes labelnumbers 
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#Vectorize resume text using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# splitting into train and test 
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.2, random_state=42)

#random forest model training
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

#evaluation process
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#  Save model files


joblib.dump(model, "models/random_forest_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("\nmodel,vectorizer,and label encoder savedin /models/")
