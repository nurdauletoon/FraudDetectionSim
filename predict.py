import pandas as pd
import joblib
import sys
import json
import numpy as np

model, scaler = joblib.load("fraud_model.pkl")

# AnyLogic передаёт путь к файлу CSV как аргумент
input_file = sys.argv[1]
df = pd.read_csv(input_file)

X = df[["amount", "hour", "merchant_risk", "device_score"]]
X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)

# Сохраняем результаты в тот же файл
df["is_fraud_pred"] = preds
df.to_csv(input_file.replace(".csv", "_pred.csv"), index=False)
print("✅ Предсказание записано:", input_file.replace(".csv", "_pred.csv"))
