# ================================
# CLEAN SETUP (NO WARNINGS)
# ================================
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# ================================
# IMPORTS
# ================================
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from scipy.spatial.distance import jensenshannon

# ================================
# GENERATE DATA
# ================================
def generate_data(n_samples=2000, timesteps=30, features=5):
    X = np.random.rand(n_samples, timesteps, features)
    y = np.random.rand(n_samples) * 100
    return X, y

# ================================
# BUILD MODEL
# ================================
def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')
    return model

# ================================
# TRAIN MODEL
# ================================
def train_model(X, y):
    model = build_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # Build once (prevents retracing)
    model.predict(X[:1], verbose=0)

    return model

# ================================
# GET ACTIVATIONS (FIXED)
# ================================
def get_activations(model, X):
    # 🔥 USE FIRST LSTM (layer index 1)
    intermediate_model = Model(
        inputs=model.inputs,
        outputs=model.layers[1].output
    )
    return intermediate_model.predict(X, verbose=0)

# ================================
# COMPUTE DRIFT
# ================================
def compute_drift(base_act, curr_act):
    b = base_act.flatten()
    c = curr_act.flatten()

    bins = 30

    b_hist, bin_edges = np.histogram(b, bins=bins, density=True)
    c_hist, _ = np.histogram(c, bins=bin_edges, density=True)

    b_hist /= (b_hist.sum() + 1e-8)
    c_hist /= (c_hist.sum() + 1e-8)

    return jensenshannon(b_hist, c_hist)

# ================================
# MAIN PIPELINE
# ================================
def main():
    print("Generating data...")
    X, y = generate_data()

    # =========================
    # TRAIN / BASE SPLIT
    # =========================
    split = int(0.7 * len(X))
    X_base, y_base = X[:split], y[:split]
    X_stream = X[split:].copy()

    print("Training model...")
    model = train_model(X_base, y_base)

    print("Extracting baseline activations...")
    base_act = get_activations(model, X_base)

    # =========================
    # CONTROLLED BATCHING (FIXED)
    # =========================
    num_batches = 5
    batch_size = len(X_stream) // num_batches

    drift_scores = []

    print("\nSimulating real-time drift...\n")

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = start + batch_size

        batch = X_stream[start:end].copy()

        # =========================
        # GRADUAL DRIFT (SMOOTH)
        # =========================
        drift_strength = (batch_idx / num_batches) * 2   # clean progression

        batch[:, :, 0] += drift_strength        # sensor drift
        batch *= (1 + drift_strength * 0.5)     # scaling drift (controlled)

        # =========================
        # ACTIVATIONS + DRIFT
        # =========================
        curr_act = get_activations(model, batch)

        drift = compute_drift(base_act, curr_act)
        drift_scores.append(drift)

        print(f"Batch {batch_idx + 1} → Drift: {drift:.4f}")

    # =========================
    # SAVE RESULTS
    # =========================
    df = pd.DataFrame({
        "batch": np.arange(1, num_batches + 1),
        "drift_score": drift_scores
    })

    df.to_csv("drift_results.csv", index=False)

    print("\n✅ Drift results saved (clean, structured)")
# ================================
if __name__ == "__main__":
    main()