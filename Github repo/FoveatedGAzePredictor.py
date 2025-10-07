import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model # <-- UPDATED: Added load_model for verification
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt 

# --- Configuration ---
# The number of previous gaze points (timesteps) the model uses to predict the next one.
SEQUENCE_LENGTH = 10
# The number of data points to generate for demonstration.
DATA_POINTS = 1000 
# Normalized coordinates range (0 to 1, representing the screen)
MIN_COORD, MAX_COORD = 0.0, 1.0 
# Filename for the exported model artifact
MODEL_FILENAME = 'gaze_predictor_lstm.h5' 

# --- Data Generation Function (Enhanced for Realism) ---

def generate_gaze_data(num_points, seq_length):
    """
    Generates synthetic 2D gaze data (x, y) sequences simulating smooth pursuits and saccades.
    """
    print(f"Generating {num_points} data points with sequence length {seq_length}...")
    
    gaze_x, gaze_y = [], []
    current_x = np.random.uniform(MIN_COORD, MAX_COORD)
    current_y = np.random.uniform(MIN_COORD, MAX_COORD)
    
    # Simulate movement over time
    for _ in range(num_points):
        # Introduce Saccades (sudden, large jump) 10% of the time
        if np.random.rand() < 0.1:
            target_x = np.random.uniform(MIN_COORD, MAX_COORD)
            target_y = np.random.uniform(MIN_COORD, MAX_COORD)
        # Smooth Pursuit (small, controlled movement) 90% of the time
        else:
            # Random walk with a slight bias
            target_x = current_x + np.random.normal(0, 0.02) 
            target_y = current_y + np.random.normal(0, 0.02)
        
        # Smoothly interpolate towards the target (simulating smooth pursuit)
        current_x = current_x * 0.9 + target_x * 0.1
        current_y = current_y * 0.9 + target_y * 0.1
        
        # Apply bounds
        current_x = np.clip(current_x, MIN_COORD, MAX_COORD)
        current_y = np.clip(current_y, MIN_COORD, MAX_COORD)
        
        gaze_x.append(current_x)
        gaze_y.append(current_y)
        
    gaze_trace = np.array([gaze_x, gaze_y]).T # Shape: (num_points, 2)

    X, y = [], []
    
    # Convert the trace into supervised learning sequences
    for i in range(len(gaze_trace) - seq_length):
        X.append(gaze_trace[i:i + seq_length]) 
        y.append(gaze_trace[i + seq_length])
    
    return np.array(X), np.array(y), gaze_trace

# --- Model Definition (OPTIMIZED STACKED LSTM) ---

def build_lstm_model(seq_length, n_features):
    """
    Defines a STACKED LSTM model for sequence prediction, allowing for
    deeper learning of complex time-series patterns (Model Optimization).
    """
    model = Sequential([
        # 1. First LSTM Layer: Must return sequences for the next LSTM layer.
        LSTM(units=100, activation='relu', input_shape=(seq_length, n_features), return_sequences=True), 
        
        # 2. Second LSTM Layer: Learns deeper, non-linear relationships.
        LSTM(units=50, activation='relu'), 
        
        # Dropout helps prevent overfitting to the synthetic data
        Dropout(0.3), 
        
        # Dense layer outputs the final prediction (2 values: x_next, y_next)
        Dense(n_features) 
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    return model

# --- Visualization Function (No changes needed here) ---

def plot_prediction_demo(input_sequence, predicted_point, actual_next_point, trace):
    """
    Plots the overall gaze trace, the current input sequence, the actual next point,
    and the model's prediction.
    """
    plt.figure(figsize=(10, 8))
    
    # 1. Plot the entire simulated gaze trace (The Ground Truth)
    plt.plot(trace[:, 0], trace[:, 1], label='Full Gaze Trace (Ground Truth)', color='#4c51bf', alpha=0.3)
    
    # 2. Plot the 10 input points used for prediction (The Current Sequence)
    seq = input_sequence[0]
    plt.scatter(seq[:, 0], seq[:, 1], label=f'Input Sequence ({SEQUENCE_LENGTH} points)', color='#f6ad55', s=50, zorder=5)
    plt.plot(seq[:, 0], seq[:, 1], color='#f6ad55', linestyle='--', linewidth=1, alpha=0.7)
    
    # 3. Plot the Actual Next Point (The Target)
    plt.scatter(actual_next_point[0], actual_next_point[1], 
                label='Actual Next Point', color='#ff0000', marker='X', s=200, zorder=10)
    
    # 4. Plot the Predicted Next Point (The Model's Guess)
    plt.scatter(predicted_point[0], predicted_point[1], 
                label='Predicted Point', color='#00ff00', marker='o', s=100, zorder=10)
    
    # 5. Connect the prediction to the last input point
    plt.plot([seq[-1, 0], predicted_point[0]], [seq[-1, 1], predicted_point[1]], 
             color='#00ff00', linestyle=':', linewidth=2, label='Predicted Vector')
    
    plt.title('Foveated Gaze Prediction Demonstration (Stacked LSTM)')
    plt.xlabel('Normalized X Coordinate')
    plt.ylabel('Normalized Y Coordinate')
    plt.xlim(MIN_COORD, MAX_COORD)
    plt.ylim(MIN_COORD, MAX_COORD)
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Prepare Data
    X_data, y_data, full_trace = generate_gaze_data(DATA_POINTS, SEQUENCE_LENGTH)
    n_features = X_data.shape[2] 
    
    # Simple train/test split (80% train, 20% test)
    split_index = int(0.8 * len(X_data))
    X_train, X_test = X_data[:split_index], X_data[split_index:]
    y_train, y_test = y_data[:split_index], y_data[split_index:]

    print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print("\n--- Building and Training Stacked LSTM Model ---")

    # 2. Build Model (Now Stacked)
    model = build_lstm_model(SEQUENCE_LENGTH, n_features)
    
    # 3. Train Model (Increased epochs for deeper model)
    history = model.fit(
        X_train, 
        y_train, 
        epochs=50, # Increased epochs for better learning
        batch_size=32, 
        verbose=0 
    )
    print("Model Training Complete (50 epochs).")

    # 4. Evaluate Model Performance
    mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Set Mean Squared Error (MSE): {mse:.6f}")
    
    # 5. Demonstration: Predict the next gaze point
    input_sequence = X_test[0:1] 
    actual_next_point = y_test[0]
    predicted_point = model.predict(input_sequence, verbose=0)[0]

    print("\n--- Prediction Demonstration Results ---")
    print(f"Actual Next Gaze Point (X, Y): [{actual_next_point[0]:.4f}, {actual_next_point[1]:.4f}]")
    print(f"Predicted Gaze Point (X, Y): [{predicted_point[0]:.4f}, {predicted_point[1]:.4f}]")
    
    # Calculate Euclidean distance error (how far off the prediction was)
    error_distance = np.linalg.norm(predicted_point - actual_next_point)
    print(f"Euclidean Prediction Error: {error_distance:.4f} (This is your key artifact value)")
    
    # 6. Save the Trained Model (ARTIFACT EXPORT)
    try:
        model.save(MODEL_FILENAME)
        print(f"\nModel saved successfully as: {MODEL_FILENAME}")
        
        # 7. Verification: Load the model and ensure it predicts the same thing
        loaded_model = load_model(MODEL_FILENAME)
        loaded_prediction = loaded_model.predict(input_sequence, verbose=0)[0]

        if np.allclose(predicted_point, loaded_prediction, atol=1e-6):
            print("✅ Verification successful: Loaded model prediction matches original prediction.")
        else:
            print("❌ Verification FAILED: Loaded model prediction MISMATCHES original prediction.")

    except Exception as e:
        print(f"\n❌ ERROR saving/loading model: {e}")


    # 8. Visualize the Prediction
    plot_prediction_demo(input_sequence, predicted_point, actual_next_point, full_trace)
    
    print("\n✅ Foveated Gaze Predictor completed: Stacked LSTM + Model Export ready for VR engine integration.")






