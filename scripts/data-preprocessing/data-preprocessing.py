import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Set paths relative to project structure
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
CLEANED_DATA_PATH = os.path.join(PROJECT_ROOT, 'scripts', 'clean-missing-data', 'performance_data_cleaned.csv')
OUTPUT_DIR = os.path.dirname(__file__)  # Save in data-preprocessing folder

print("="*60)
print("STEP 1: DATA NORMALIZATION")
print("="*60)

# ============================================
# LOAD CLEANED DATA
# ============================================
df = pd.read_csv(CLEANED_DATA_PATH)
print(f"\n1. Loaded cleaned data: {len(df)} websites")

# ============================================
# DEFINE METRICS TO NORMALIZE
# ============================================
# All numeric performance metrics
metrics = [
    'LCP_ms', 'FCP_ms', 'TBT_ms', 'TTI_ms', 'SpeedIndex_ms',
    'TTFB_ms', 'noOfRequests', 'pageSize_kb', 
    'javascriptExecution_ms', 'loadTime_ms'
]

print(f"\n2. Metrics to normalize: {len(metrics)}")
for m in metrics:
    print(f"   - {m}")

# ============================================
# CHECK ORIGINAL RANGES (BEFORE NORMALIZATION)
# ============================================
print(f"\n3. Original metric ranges:")
desc = df[metrics].describe()
print(desc.loc[['min', 'max']])

# ============================================
# NORMALIZE TO [0, 1] RANGE
# ============================================
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[metrics] = scaler.fit_transform(df[metrics])

print(f"\n4. ✓ Normalization complete")
print(f"   All metrics now in range [0, 1]")

# Verify normalization
print(f"\n5. Normalized ranges (should be 0 to 1):")
print(df_normalized[metrics].describe().loc[['min', 'max']])

# ============================================
# INVERT METRICS (Lower is better → Higher is better)
# ============================================
# For these metrics, LOWER values = BETTER performance
# After inversion: HIGHER values = BETTER performance
invert_metrics = [
    'LCP_ms',           # Lower LCP = faster loading
    'FCP_ms',           # Lower FCP = faster first paint
    'TBT_ms',           # Lower TBT = less blocking
    'TTI_ms',           # Lower TTI = faster interactive
    'SpeedIndex_ms',    # Lower SI = faster visual load
    'TTFB_ms',          # Lower TTFB = faster server
    'javascriptExecution_ms',  # Lower JS = less processing
    'loadTime_ms',      # Lower load = faster overall
    'noOfRequests',     # Fewer requests = better
    'pageSize_kb'       # Smaller size = better
]

print(f"\n6. Inverting metrics (lower=better → higher=better):")
for metric in invert_metrics:
    df_normalized[metric] = 1 - df_normalized[metric]
    print(f"   ✓ Inverted {metric}")

# ============================================
# VERIFY INVERSION
# ============================================
print(f"\n7. Sample normalized & inverted values:")
print(f"   (Higher values = better performance)")
print(df_normalized[['requested_url'] + metrics[:5]].head(10))

# ============================================
# SAVE PREPROCESSED DATA AND SCALER
# ============================================
df_normalized.to_csv(os.path.join(OUTPUT_DIR, 'data_preprocessed.csv'), index=False)
print(f"\n8. ✓ Saved: data_preprocessed.csv")

# Save scaler for future use (important!)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
print(f"   ✓ Saved: scaler.pkl (for future predictions)")

# ============================================
# SUMMARY
# ============================================
print(f"\n" + "="*60)
print("NORMALIZATION SUMMARY")
print("="*60)
print(f"✓ Normalized {len(metrics)} metrics to [0, 1]")
print(f"✓ Inverted {len(invert_metrics)} metrics")
print(f"✓ Higher values now = better performance")
print(f"✓ Ready for target score creation")
print("="*60)