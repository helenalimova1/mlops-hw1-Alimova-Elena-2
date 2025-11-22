
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os
import yaml

# Загрузка параметров
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Загрузка подготовленных данных
df = pd.read_csv("data/processed/train.csv")
X = df.drop("label", axis=1)
y = df["label"]

# Обучение модели
model = LogisticRegression(max_iter=300)
model.fit(X, y)

# Сохранение модели
joblib.dump(model, "models/model.pkl")
print("Сохранена модель в models/model.pkl")
