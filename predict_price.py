import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load saved model and scaler
def load_model():
    with open('laptop_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    with open('feature_columns.pkl', 'rb') as file:
        feature_columns = pickle.load(file)
    
    return model, scaler, feature_columns

def preprocess_input(user_data, feature_columns):
    """Preprocess user input to match training data format"""
    
    # Convert to DataFrame
    df = pd.DataFrame([user_data])
    
    # Label Encoding for ordinal features
    cpu_line_mapping = {
        'Core i3': 3, 'Core i5': 5, 'Core i7': 7, 'Core i9': 9,
        'Ryzen 3': 3, 'Ryzen 5': 5, 'Ryzen 7': 7, 'Ryzen 9': 9,
        'Pentium': 2, 'Celeron': 1, 'Xeon': 8, 'Core M': 4, 'Atom': 1,
        'A4-Series': 1.5, 'A6-Series': 2, 'A8-Series': 2.5,
        'A9-Series': 3, 'A10-Series': 3.5, 'A12-Series': 4,
        'E-Series': 1, 'Unknown': 3
    }
    df['cpu_line'] = df['cpu_line'].map(cpu_line_mapping).fillna(3)
    
    cpu_type_mapping = {
        'HK': 6, 'HQ': 5, 'H': 4, 'HS': 3, 'U': 2, 
        'Y': 1, 'M': 2, 'T': 2, 'Unknown': 2
    }
    df['cpu_type_suffix'] = df['cpu_type_suffix'].map(cpu_type_mapping).fillna(2)
    
    resolution_mapping = {
        'Standard': 1, 'Full HD': 2, 'Quad HD': 3, 
        'Quad HD+': 4, '4K Ultra HD': 5
    }
    df['resolution_type'] = df['resolution_type'].map(resolution_mapping).fillna(1)
    
    # Encode gpu_model
    le = LabelEncoder()
    if 'gpu_model' in df.columns:
        df['gpu_model'] = le.fit_transform(df['gpu_model'].astype(str))
    
    # One-Hot Encoding
    cols_to_encode = ['Company', 'TypeName', 'OpSys', 'cpu_company', 'gpu_company', 'gpu_series']
    df = pd.get_dummies(df, columns=[col for col in cols_to_encode if col in df.columns], drop_first=True)
    
    # Ensure all training columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Keep only training columns in same order
    df = df[feature_columns]
    
    return df

def predict_price(user_data):
    """Make price prediction from user input"""
    
    # Load model
    model, scaler, feature_columns = load_model()
    
    # Preprocess input
    processed_data = preprocess_input(user_data, feature_columns)
    
    # Scale the data
    processed_data_scaled = scaler.transform(processed_data)
    
    # Make prediction
    prediction = model.predict(processed_data_scaled)[0]
    
    # Return prediction directly - no multipliers needed
    # The model has been trained on merged dataset with modern hardware
    return prediction

# Test the function
if __name__ == "__main__":
    test_input = {
        'Company': 'Dell',
        'TypeName': 'Notebook',
        'Inches': 15.6,
        'Ram': 8,
        'Weight': 2.5,
        'OpSys': 'Windows 10',
        # CPU features
        'cpu_company': 'Intel',
        'cpu_line': 'Core i5',
        'cpu_generation': 8,
        'cpu_type_suffix': 'U',
        'cpu_clock_speed': 1.6,
        # Screen features
        'resolution_type': 'Full HD',
        'resolution_width': 1920,
        'resolution_height': 1080,
        'touchscreen': 0,
        'ips_panel': 0,
        'retina_display': 0,
        # GPU features
        'gpu_company': 'Intel',
        'gpu_series': 'HD Graphics',
        'gpu_model': '620',
        # Storage
        'HDD': 0,
        'SSD': 256,
        'Hybrid': 0,
        'Flash_Storage': 0
    }
    
    try:
        price = predict_price(test_input)
        print(f"Predicted Price: ₹{price:,.2f}")
    except Exception as e:
        print(f"Error: {e}")