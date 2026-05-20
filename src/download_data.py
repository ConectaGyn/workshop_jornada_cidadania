import os
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUTPUT_PATH = "data/raw/telco_churn.csv"

os.makedirs("data/raw", exist_ok=True)

df = pd.read_csv(DATA_URL)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Dataset salvo em {OUTPUT_PATH} ({len(df)} linhas, {df.shape[1]} colunas)")
