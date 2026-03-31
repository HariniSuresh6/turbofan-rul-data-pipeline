import pandas as pd
from sqlalchemy import create_engine

# DB connection
DB_URI = "postgresql://admin:admin@localhost:5432/turbofan_db"

def load_parquet():
    print("Loading Parquet file...")
    df = pd.read_parquet("data/raw/engine_data.parquet")
    print(f"Loaded {df.shape[0]} rows")
    return df

def load_to_postgres(df):
    print("Connecting to PostgreSQL...")
    engine = create_engine(DB_URI)

    print("Writing data to warehouse table: engine_sensor_data...")
    df.to_sql(
        "engine_sensor_data",
        engine,
        if_exists="replace",
        index=False,
        chunksize=10000
    )

    print("✅ Data successfully loaded into PostgreSQL!")

def main():
    df = load_parquet()
    load_to_postgres(df)

if __name__ == "__main__":
    main()