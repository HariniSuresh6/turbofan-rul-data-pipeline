import pandas as pd
df = pd.read_parquet("data/raw/engine_data.parquet")
print(df.head())