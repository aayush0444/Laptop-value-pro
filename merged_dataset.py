"""
IMPROVED DATASET MERGER WITH PROPER PREPROCESSING
==================================================
This version handles price normalization and outliers better
"""

import pandas as pd
import numpy as np
import re

# ==================== STEP 1: LOAD DATASETS ====================

print("Loading datasets...")
old_data = pd.read_csv('laptop_data.csv')
new_data = pd.read_csv('new_data_set.csv')

print(f"Old dataset: {old_data.shape}")
print(f"New dataset: {new_data.shape}")

# ==================== STEP 2: CLEAN OLD DATASET ====================

if 'Unnamed: 0' in old_data.columns:
    old_data = old_data.drop('Unnamed: 0', axis=1)

print("\nOld dataset price stats (INR):")
print(old_data['Price'].describe())

# ==================== STEP 3: STANDARDIZE NEW DATASET ====================

print("\n" + "="*70)
print("Standardizing new dataset...")
print("="*70)

new_data_standardized = pd.DataFrame()

# Basic columns
new_data_standardized['Company'] = new_data['Company']
new_data_standardized['TypeName'] = new_data['TypeName']
new_data_standardized['Inches'] = new_data['Inches']
new_data_standardized['ScreenResolution'] = new_data['ScreenResolution']

# Reconstruct CPU
def reconstruct_cpu(row):
    company = str(row['CPU_Company'])
    cpu_type = str(row['CPU_Type'])
    freq = str(row['CPU_Frequency (GHz)'])
    return f"{company} {cpu_type} {freq}GHz"

new_data_standardized['Cpu'] = new_data.apply(reconstruct_cpu, axis=1)
new_data_standardized['Ram'] = new_data['RAM (GB)'].astype(str) + 'GB'
new_data_standardized['Memory'] = new_data['Memory']

# Reconstruct GPU
def reconstruct_gpu(row):
    company = str(row['GPU_Company'])
    gpu_type = str(row['GPU_Type'])
    return f"{company} {gpu_type}"

new_data_standardized['Gpu'] = new_data.apply(reconstruct_gpu, axis=1)
new_data_standardized['OpSys'] = new_data['OpSys']
new_data_standardized['Weight'] = new_data['Weight (kg)'].astype(str) + 'kg'

# ==================== CRITICAL: BETTER PRICE CONVERSION ====================

print("\n" + "="*70)
print("IMPROVED PRICE CONVERSION")
print("="*70)

# Check current EUR to INR rate (as of 2024-2025)
# Using more accurate conversion: 1 EUR ≈ 92 INR
EURO_TO_INR = 92.0

# Convert prices
new_data_standardized['Price'] = new_data['Price (Euro)'] * EURO_TO_INR

print(f"\nNew dataset price stats BEFORE normalization:")
print(new_data_standardized['Price'].describe())

# Compare price distributions
print(f"\nOld data mean: ₹{old_data['Price'].mean():,.0f}")
print(f"New data mean: ₹{new_data_standardized['Price'].mean():,.0f}")

# Check if prices need adjustment
old_mean = old_data['Price'].mean()
new_mean = new_data_standardized['Price'].mean()

# If new data is significantly different, normalize it
price_ratio = old_mean / new_mean
print(f"\nPrice ratio (old/new): {price_ratio:.2f}")

if 0.7 < price_ratio < 1.3:
    print("✅ Price distributions are similar, no adjustment needed")
else:
    print(f"⚠️  Price distributions differ significantly")
    print(f"Applying normalization factor: {price_ratio:.2f}")
    new_data_standardized['Price'] = new_data_standardized['Price'] * price_ratio

print(f"\nNew data mean AFTER adjustment: ₹{new_data_standardized['Price'].mean():,.0f}")

# ==================== STEP 4: MERGE DATASETS ====================

print("\n" + "="*70)
print("Merging datasets...")
print("="*70)

combined_data = pd.concat([old_data, new_data_standardized], axis=0, ignore_index=True)

print(f"Combined dataset shape: {combined_data.shape}")

# Remove duplicates
duplicates = combined_data.duplicated().sum()
if duplicates > 0:
    print(f"Removing {duplicates} duplicates...")
    combined_data = combined_data.drop_duplicates()

# ==================== STEP 5: OUTLIER REMOVAL ====================

print("\n" + "="*70)
print("OUTLIER REMOVAL")
print("="*70)

print(f"\nBefore outlier removal: {len(combined_data)} laptops")
print(f"Price range: ₹{combined_data['Price'].min():,.0f} - ₹{combined_data['Price'].max():,.0f}")

# Remove extreme outliers using IQR method
Q1 = combined_data['Price'].quantile(0.25)
Q3 = combined_data['Price'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nIQR bounds: ₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}")

# Keep data within bounds
combined_data = combined_data[
    (combined_data['Price'] >= lower_bound) & 
    (combined_data['Price'] <= upper_bound)
]

print(f"After outlier removal: {len(combined_data)} laptops")
print(f"Price range: ₹{combined_data['Price'].min():,.0f} - ₹{combined_data['Price'].max():,.0f}")

# Additional sanity check - remove unrealistic prices
combined_data = combined_data[
    (combined_data['Price'] >= 10000) &  # Min ₹10k
    (combined_data['Price'] <= 500000)   # Max ₹5 lakh
]

print(f"After sanity check: {len(combined_data)} laptops")

# ==================== STEP 6: DATA QUALITY IMPROVEMENTS ====================

print("\n" + "="*70)
print("DATA QUALITY CHECKS")
print("="*70)

# Handle missing values
print(f"\nMissing values:")
print(combined_data.isnull().sum()[combined_data.isnull().sum() > 0])

# Fill missing values if any
combined_data = combined_data.dropna()

print(f"After removing NaN: {len(combined_data)} laptops")

# ==================== STEP 7: FINAL STATISTICS ====================

print("\n" + "="*70)
print("FINAL DATASET STATISTICS")
print("="*70)

print(f"\nTotal laptops: {len(combined_data)}")
print(f"\nPrice statistics:")
print(combined_data['Price'].describe())

print(f"\nCompany distribution:")
print(combined_data['Company'].value_counts().head(10))

print(f"\nType distribution:")
print(combined_data['TypeName'].value_counts())

# Analyze CPU generations
def extract_cpu_gen(cpu_string):
    if pd.isna(cpu_string):
        return None
    intel_match = re.search(r'i[3579]\s*(\d)(\d{3})', str(cpu_string))
    if intel_match:
        return int(intel_match.group(1))
    ryzen_match = re.search(r'Ryzen.*?(\d)(\d{3})', str(cpu_string))
    if ryzen_match:
        return int(ryzen_match.group(1))
    return None

combined_data['CPU_Gen_Temp'] = combined_data['Cpu'].apply(extract_cpu_gen)
print(f"\nCPU Generation coverage:")
for gen in sorted(combined_data['CPU_Gen_Temp'].dropna().unique()):
    count = len(combined_data[combined_data['CPU_Gen_Temp'] == gen])
    print(f"  {int(gen)}th gen: {count} laptops")

combined_data = combined_data.drop('CPU_Gen_Temp', axis=1)

# ==================== STEP 8: SAVE ====================

print("\n" + "="*70)
print("SAVING CLEANED DATASET")
print("="*70)

combined_data.to_csv('laptop_data_merged_clean.csv', index=False)
print(f"✅ Saved as 'laptop_data_merged_clean.csv'")

print("\n" + "="*70)
print("✅ DATASET PREPARATION COMPLETE!")
print("="*70)

print("\n📋 NEXT STEPS:")
print("1. Use 'laptop_data_merged_clean.csv' in your EDA notebook")
print("2. The prices are now normalized and outliers removed")
print("3. Expected model performance should improve significantly")
print(f"4. Training on {len(combined_data)} high-quality laptop entries")

print("\n💡 RECOMMENDED EDA IMPROVEMENTS:")
print("1. Add feature scaling before training")
print("2. Try polynomial features for better fit")
print("3. Use GridSearchCV for hyperparameter tuning")
print("4. Consider ensemble methods (Stacking)")

print("\n" + "="*70)