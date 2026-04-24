import pandas as pd
from sqlalchemy import create_engine
import os

# IMPORTANT: use docker service name
DB_URI = "postgresql://admin:admin@localhost:5432/turbofan_db"
def load_data():
    print("Loading data from warehouse...")
    engine = create_engine(DB_URI)

    df = pd.read_sql("SELECT * FROM engine_sensor_data", engine)
    print(f"Loaded {df.shape[0]} rows")

    return df

def create_rul(df):
    print("Calculating RUL...")

    df['max_cycle'] = df.groupby('engine_id')['cycle'].transform('max')
    df['RUL'] = df['max_cycle'] - df['cycle']

    # feature
    df['cycle_norm'] = df['cycle'] / df['max_cycle']

    df.drop(columns=['max_cycle'], inplace=True)

    return df

def save_processed(df):
    print("Saving processed data...")

    os.makedirs("data/processed", exist_ok=True)

    # save to data lake
    df.to_parquet("data/processed/engine_features.parquet", index=False)

    # save to warehouse
    engine = create_engine(DB_URI)

    df.to_sql(
        "engine_features",
        engine,
        if_exists="replace",
        index=False,
        chunksize=10000
    )

    print("✅ Processed data saved!")

def main():
    df = load_data()
    df = create_rul(df)
    save_processed(df)

if __name__ == "__main__":
    main()