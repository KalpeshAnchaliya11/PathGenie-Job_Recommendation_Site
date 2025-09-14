import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

from sklearn.preprocessing import label_binarize


import joblib
import pandas as pd

# loading model and testdata
model = joblib.load("models/random_forest_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

label_encoder = joblib.load("models/label_encoder.pkl")

df = pd.read_csv("resume_dataset.csv")


X = vectorizer.transform(df["resume_text"])
y = label_encoder.transform(df["role"])

#splitting data for ecaluation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#confusion matrix
y_pred = model.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            

            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()

plt.savefig("confusion_matrix.png")
plt.close()
# precision recall curve (micro)
y_test_bin = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))
y_score = model.predict_proba(X_test)

precision = dict()
recall = dict()

avg_precision = dict()
for i in range(len(label_encoder.classes_)):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])


    avg_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

plt.figure(figsize=(8, 6))
for i in range(len(label_encoder.classes_)):
    plt.plot(recall[i], precision[i], label=f"{label_encoder.classes_[i]} (AP={avg_precision[i]:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")


plt.legend()
plt.tight_layout()
plt.savefig("precision_recall_curve.png")
plt.close()

print("dashboard chart saved: confusion_matrix.png, precision_recall_curve.png")
