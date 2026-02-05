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
    
    # Apply price multipliers for hardware not in training data (2017-2018 laptops)
    price_multiplier = 1.0
    
    # CPU Generation multipliers (training data: mostly 6th-8th gen)
    cpu_gen = user_data.get('cpu_generation', 7)
    if cpu_gen >= 9 and cpu_gen <= 10:
        price_multiplier *= 1.02  # 9th-10th gen: +2%
    elif cpu_gen == 11:
        price_multiplier *= 1.03# 11th gen: +3%
    elif cpu_gen == 12:
        price_multiplier *= 1.05  # 12th gen: +5%
    elif cpu_gen == 13:
        price_multiplier *= 1.09  # 13th gen: +9%
    elif cpu_gen >= 14:
        price_multiplier *= 1.13  # 14th gen+: +13%
    
    # CPU Line multipliers (i9 and Ryzen 9 premium)
    cpu_line = user_data.get('cpu_line', '')
    if cpu_line in ['Core i9', 'Ryzen 9']:
        price_multiplier *= 1.10  # Flagship processors: +10%
    elif cpu_line in ['Core i7', 'Ryzen 7']:
        price_multiplier *= 1.05  # High-performance: +5%
    # i5/Ryzen 5 and below get no multiplier (mainstream)
    
    # GPU Series multipliers (training data: mostly GTX 10 series)
    gpu_series = user_data.get('gpu_series', '')
    gpu_company = user_data.get('gpu_company', '')
    
    # Nvidia RTX series (newer than training data)
    if 'RTX 40' in gpu_series:
        # Check specific models for RTX 40
        gpu_model = str(user_data.get('gpu_model', '')).lower()
        if any(x in gpu_model for x in ['4090', '4080']):
            price_multiplier *= 1.20  # High-end RTX 40: +35%
        elif any(x in gpu_model for x in ['4070', '4060']):
            price_multiplier *= 1.13  # Mid-range RTX 40: +20%
        else:
            price_multiplier *= 1.10  # RTX 40 general: +25%
    
    elif 'RTX 30' in gpu_series:
        gpu_model = str(user_data.get('gpu_model', '')).lower()
        if any(x in gpu_model for x in ['3090', '3080', '3070']):
            price_multiplier *= 1.10  # High-end RTX 30: +20%
        elif any(x in gpu_model for x in ['3060', '3050']):
            price_multiplier *= 1.05  # Entry RTX 30 (3050/3060): +8%
        else:
            price_multiplier *= 1.03  # RTX 30 general: +15%
    
    
    
    # Newer GTX series
    elif 'GTX 16' in gpu_series:
        price_multiplier *= 1.05  # GTX 16 series: +5%
    
    # AMD Radeon RX newer series
    elif 'RX 7000' in gpu_series:
        price_multiplier *= 1.30  # RX 7000: +30%
    elif 'RX 6000' in gpu_series:
        price_multiplier *= 1.20  # RX 6000: +20%
    elif 'RX 5000' in gpu_series:
        price_multiplier *= 1.08  # RX 5000: +8%
    
    # Intel newer integrated graphics (minimal boost, it's still integrated)
    elif gpu_company == 'Intel':
        if 'Iris Xe' in gpu_series:
            price_multiplier *= 1.03  # Iris Xe: +3%
        elif 'UHD Graphics' in gpu_series:
            price_multiplier *= 1.02  # UHD: +2%
        # HD Graphics and older get no multiplier
    
    # High-end display multiplier (4K, Quad HD+)
    resolution = user_data.get('resolution_type', '')
    if resolution == '4K Ultra HD':
        price_multiplier *= 1.12  # 4K display: +12%
    elif resolution == 'Quad HD+':
        price_multiplier *= 1.06  # QHD+: +6%
    elif resolution == 'Quad HD':
        price_multiplier *= 1.04  # QHD: +4%
    # Full HD and Standard get no multiplier
    
    # High RAM multiplier (32GB+ wasn't common in training data)
    ram = user_data.get('Ram', 8)
    if ram >= 32:
        price_multiplier *= 1.15  # 32GB+: +15%
    elif ram == 24:
        price_multiplier *= 1.08  # 24GB: +8%
    elif ram == 16:
        price_multiplier *= 1.03  # 16GB: +3%
    # 8GB and below get no multiplier
    
    # Large SSD multiplier (1TB+ SSD was premium in 2017-2018)
    ssd = user_data.get('SSD', 0)
    if ssd >= 2048:
        price_multiplier *= 1.12  # 2TB+ SSD: +12%
    elif ssd >= 1024:
        price_multiplier *= 1.08  # 1TB SSD: +8%
    elif ssd >= 512:
        price_multiplier *= 1.04  # 512GB SSD: +4%
    # 256GB and below get no multiplier
    
    # Apply the multiplier
    adjusted_prediction = prediction * price_multiplier
    
    return adjusted_prediction

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