import pandas as pd
data = pd.read_csv("Telco-Customer-Churn.csv")
print(data.head())
print(data.shape)
print(data.info())
print(data.isnull().sum())

data = data.drop("customerID", axis=1)


data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].mean())

print(data.head())
print(data.info())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

object_columns = data.select_dtypes(include=["object"]).columns

for col in object_columns:
    data[col] = le.fit_transform(data[col])

print(data.dtypes)
from sklearn.model_selection import train_test_split

X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model Trained Successfully!")
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nFirst 10 Predictions:")
print(model.predict(X_test[:10]))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt

data["Churn"].value_counts().plot(kind="bar")

plt.title("Customer Churn Count")
plt.xlabel("Churn")
plt.ylabel("Count")

plt.show()