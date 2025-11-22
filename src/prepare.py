
import pandas as pd
import os
import yaml

# Загрузка параметров
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Загрузка данных через DVC
df = pd.read_csv("data/train.csv")

# Сплит на train/test
train = df.sample(frac=1 - params['split_ratio'], random_state=params['random_state'])
test = df.drop(train.index)

# Сохранение обработанных данных
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
print("Сохранены обработанные данные в data/processed/")
