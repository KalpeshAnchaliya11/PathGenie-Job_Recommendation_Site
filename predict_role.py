import joblib

import os
# load mdel components only once
model =joblib.load(os.path.join("models", "random_forest_model.pkl"))
vectorizer= joblib.load(os.path.join("models", "vectorizer.pkl"))
label_encoder= joblib.load(os.path.join("models", "label_encoder.pkl"))

# Predict job role from resume text
def predict_role_from_text(text):
    vector=vectorizer.transform([text])


    prediction=model.predict(vector)
    role=label_encoder.inverse_transform(prediction)[0]
    return role
