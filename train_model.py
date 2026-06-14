import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv("student_data.csv")

X = data[
    ["StudyHours",
     "Attendance",
     "PreviousMarks"]
]

y = data["FinalScore"]

model = LinearRegression()

model.fit(X, y)

joblib.dump(
    model,
    "model.pkl"
)

print("Model Trained Successfully")