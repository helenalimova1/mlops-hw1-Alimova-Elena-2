
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml

# Загрузка параметров
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Загрузка данных
df = pd.read_csv("data/raw/train.csv")

# Сплит на train/test
train, test = train_test_split(df, test_size=params['split_ratio'], random_state=params['random_state'])

# Сохранение обработанных данных
os.makedirs("data/processed", exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
print("Сохранены обработанные данные в data/processed/")
