import pandas as pd
import os
import kagglehub

# Define columns
columns = ['engine_id', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

def download_dataset():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("behrad3d/nasa-cmaps")
    print(f"Dataset downloaded to: {path}")
    return path

def find_file(base_path, filename):
    for root, dirs, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found in dataset")

def load_data(file_path):
    df = pd.read_csv(file_path, sep=r'\s+', header=None)
    df = df.iloc[:, :len(columns)]
    df.columns = columns
    return df

def save_parquet(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)

def main():
    # Step 1: Download dataset
    dataset_path = download_dataset()

    # Step 2: Locate correct file
    train_file = find_file(dataset_path, "train_FD002.txt")

    print(f"Using file: {train_file}")

    # Step 3: Load data
    print("Loading dataset...")
    df = load_data(train_file)

    # Step 4: Validate
    print(f"Dataset shape: {df.shape}")
    print(f"Unique engines: {df['engine_id'].nunique()}")

    # Step 5: Save to data lake
    output_path = "data/raw/engine_data.parquet"
    print("Saving as Parquet...")
    save_parquet(df, output_path)

    print("✅ Ingestion completed successfully!")

if __name__ == "__main__":
    main()