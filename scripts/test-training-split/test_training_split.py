import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import json

print("="*60)
print("STEP 3: TRAIN/TEST SPLIT (CORRECTED)")
print("="*60)

# ============================================
# LOAD DATA WITH TARGET (CORRECTED VERSION)
# ============================================
print("\n1. Loading corrected data with balanced quartiles...")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
CURRENT_DIR = os.path.dirname(__file__)
csv_file = os.path.join(PROJECT_ROOT, 'scripts', 'single-target-performance', 'data_with_target_corrected.csv')
df = pd.read_csv(csv_file)
print(f"   Loaded: {len(df)} websites")

# ============================================
# CHECK IF QUARTILE COLUMN EXISTS
# ============================================
if 'category_quantile' not in df.columns:
    print("\n   ⚠️  WARNING: No quartile column found!")
    print("   Creating quartile bins now...")
    
    # Create quartiles
    quartiles = df['target'].quantile([0, 0.25, 0.5, 0.75, 1.0])
    labels = ['Q1 (Bottom 25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (Top 25%)']
    df['category_quantile'] = pd.cut(df['target'], bins=quartiles, 
                                      labels=labels, include_lowest=True, 
                                      duplicates='drop')

print(f"\n   Quartile distribution:")
print(df['category_quantile'].value_counts().sort_index())

# ============================================
# DEFINE FEATURES AND TARGET
# ============================================


# IMPORTANT: Include ALL metrics used in target creation for now
# (You'll need to address data leakage later with derived features)
feature_columns = [
    'LCP_ms', 'FCP_ms', 'TBT_ms', 'CLS', 'TTI_ms', 'SpeedIndex_ms',
    'TTFB_ms', 'noOfRequests', 'pageSize_kb', 
    'javascriptExecution_ms', 'loadTime_ms'
]


# Add derived features if they exist (from improved script)
derived_features = ['size_per_request', 'lcp_fcp_ratio', 'loading_efficiency', 
                   'interactivity_score', 'js_intensity']
for feature in derived_features:
    if feature in df.columns:
        feature_columns.append(feature)

X = df[feature_columns]
y = df['target']
stratify_column = df['category_quantile']

print(f"   Features: {len(feature_columns)}")
print(f"   Feature list: {', '.join(feature_columns[:5])}... ({len(feature_columns)} total)")
print(f"   Target: 'target' (performance score)")
print(f"   Stratify by: 'category_quantile'")

# ============================================
# CRITICAL WARNING ABOUT DATA LEAKAGE
# ============================================
print(f"\n   ⚠️  DATA LEAKAGE WARNING:")
print(f"   Your target was created from these features!")
print(f"   This will cause artificially high accuracy.")
print(f"   For final research, you should:")
print(f"   1. Use external metric (Lighthouse score), OR")
print(f"   2. Predict one metric from others, OR")
print(f"   3. Use only derived features")

# ============================================
# STRATIFIED SPLIT: 70% TRAIN, 30% TEST
# ============================================
print(f"\n3. Performing STRATIFIED split (70/30)...")

X_train, X_test, y_train, y_test, strat_train, strat_test = train_test_split(
    X, y, stratify_column,
    test_size=0.3,           # 30% for testing
    random_state=42,         # Reproducible results
    stratify=stratify_column # CRITICAL: Maintain quartile distribution!
)

print(f"\n   Split completed:")
print(f"   Training set:   {len(X_train)} websites ({len(X_train)/len(df)*100:.1f}%)")
print(f"   Test set:       {len(X_test)} websites ({len(X_test)/len(df)*100:.1f}%)")

# ============================================
# VERIFY STRATIFICATION WORKED
# ============================================
print(f"\n4. Verifying stratification quality...")
print(f"\n   Overall target distribution:")
print(f"   Training - Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
print(f"   Test     - Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")
print(f"   Difference: {abs(y_train.mean() - y_test.mean()):.4f}")

print(f"\n   Quartile distribution preservation:")
print(f"   {'Quartile':<20} {'Original':<12} {'Train':<12} {'Test':<12} {'Status':<10}")

original_dist = stratify_column.value_counts(normalize=True).sort_index()
train_dist = strat_train.value_counts(normalize=True).sort_index()
test_dist = strat_test.value_counts(normalize=True).sort_index()

all_good = True
for label in original_dist.index:
    orig_pct = original_dist[label] * 100
    train_pct = train_dist[label] * 100
    test_pct = test_dist[label] * 100
    
    # Check if proportions are maintained (within 2% tolerance)
    diff = abs(train_pct - test_pct)
    status = "✓ Good" if diff < 2.0 else "⚠️ Check"
    if diff >= 2.0:
        all_good = False
    
    print(f"   {label:<20} {orig_pct:>6.1f}%      {train_pct:>6.1f}%      {test_pct:>6.1f}%      {status}")

if all_good:
    print(f"\n   ✓ Stratification successful! Distributions are balanced.")
else:
    print(f"\n   ⚠️  Warning: Some quartiles show distribution differences")

# ============================================
# CHECK SAMPLE SIZES PER QUARTILE
# ============================================
print(f"\n5. Sample sizes per quartile:")
print(f"   {'Quartile':<20} {'Train Samples':<15} {'Test Samples':<15} {'Status':<10}")
print(f"   {'-'*70}")

train_counts = strat_train.value_counts().sort_index()
test_counts = strat_test.value_counts().sort_index()

min_acceptable = 50  # Minimum for statistical validity

for label in train_counts.index:
    train_n = train_counts[label]
    test_n = test_counts[label]
    
    status = "✓ Good" if test_n >= min_acceptable else "⚠️ Low"
    
    print(f"   {label:<20} {train_n:<15} {test_n:<15} {status}")

print(f"\n   Minimum test samples: {test_counts.min()}")
if test_counts.min() >= min_acceptable:
    print(f"   ✓ All quartiles have sufficient test samples (≥{min_acceptable})")
else:
    print(f"   ⚠️  Some quartiles have insufficient test samples")

# ============================================
# SAVE SPLIT DATA
# ============================================
print(f"\n6. Saving split datasets...")

# Save features
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)

# Save targets
y_train.to_csv('y_train.csv', index=False, header=['target'])
y_test.to_csv('y_test.csv', index=False, header=['target'])

# Save stratification labels (useful for per-quartile evaluation)
strat_train.to_csv('train_quartiles.csv', index=False, header=['quartile'])
strat_test.to_csv('test_quartiles.csv', index=False, header=['quartile'])

print(f"   ✓ Saved: X_train.csv ({X_train.shape})")
print(f"   ✓ Saved: X_test.csv ({X_test.shape})")
print(f"   ✓ Saved: y_train.csv ({y_train.shape})")
print(f"   ✓ Saved: y_test.csv ({y_test.shape})")
print(f"   ✓ Saved: train_quartiles.csv")
print(f"   ✓ Saved: test_quartiles.csv")

# ============================================
# SAVE DETAILED METADATA
# ============================================
metadata = {
    'total_samples': len(df),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'train_percentage': round(len(X_train)/len(df)*100, 2),
    'test_percentage': round(len(X_test)/len(df)*100, 2),
    'features': feature_columns,
    'num_features': len(feature_columns),
    'target': 'target',
    'stratification': 'category_quantile',
    'random_state': 42,
    'stratification_verified': all_good,
    'min_test_samples_per_quartile': int(test_counts.min()),
    'quartile_distribution': {
        'train': {str(k): int(v) for k, v in train_counts.items()},
        'test': {str(k): int(v) for k, v in test_counts.items()}
    },
    'data_leakage_warning': 'Target created from features - use with caution',
    'recommended_fixes': [
        'Use external performance metric as target',
        'Predict one metric from others (exclude from features)',
        'Use only derived features (ratios, interactions)'
    ]
}

with open('split_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"   ✓ Saved: split_metadata.json")

# ============================================
# SAVE HUMAN-READABLE SUMMARY
# ============================================
with open('split_summary.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("TRAIN/TEST SPLIT SUMMARY\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Dataset: {len(df)} websites\n")
    f.write(f"Split ratio: 70% train / 30% test\n")
    f.write(f"Stratification: By performance quartiles\n\n")
    
    f.write(f"Training set: {len(X_train)} websites\n")
    f.write(f"Test set: {len(X_test)} websites\n\n")
    
    f.write("Quartile Distribution:\n")
    f.write(f"{'Quartile':<20} {'Train':<10} {'Test':<10}\n")
    f.write("-"*40 + "\n")
    for label in train_counts.index:
        f.write(f"{label:<20} {train_counts[label]:<10} {test_counts[label]:<10}\n")
    
    f.write(f"\nFeatures ({len(feature_columns)}):\n")
    for i, feat in enumerate(feature_columns, 1):
        f.write(f"{i}. {feat}\n")
    
    f.write("\n" + "="*60 + "\n")
    f.write("IMPORTANT NOTES\n")
    f.write("="*60 + "\n")
    f.write("1. Data split is STRATIFIED to maintain quartile distribution\n")
    f.write("2. Each quartile has sufficient samples for reliable evaluation\n")
    f.write("3. Random state = 42 for reproducibility\n")
    f.write("4. WARNING: Potential data leakage - target created from features\n")
    f.write("5. For final research, address data leakage issue\n")

print(f"   ✓ Saved: split_summary.txt")

# ============================================
# FINAL SUMMARY
# ============================================
print(f"\n" + "="*60)
print("TRAIN/TEST SPLIT SUMMARY")
print("="*60)
print(f"✓ Data split 70/30 with STRATIFICATION")
print(f"✓ Training: {len(X_train)} samples")
print(f"✓ Test: {len(X_test)} samples")
print(f"✓ Features: {len(feature_columns)}")
print(f"✓ Min test samples per quartile: {test_counts.min()}")
print(f"✓ Stratification verified: {'YES' if all_good else 'CHECK WARNINGS'}")
print(f"\n⚠️  REMEMBER: Address data leakage before final results!")
print(f"   See split_metadata.json for recommended fixes")
print("="*60)