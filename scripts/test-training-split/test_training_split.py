"""
Train-Test Split Script for Web Performance Classification
Following Ghattas et al. (2025) Methodology

This script implements the data splitting approach used in:
"A Novel Approach for Evaluating Web Page Performance Based on Machine Learning 
Algorithms and Optimization Algorithms" (Ghattas et al., 2025)

Key Details from Ghattas et al. (2025):
- Dataset: 1,208 balanced samples (800 original + 408 SMOTE oversampled)
- Split: 70% training (845 samples) / 30% testing (363 samples)
- Evaluation: 10-fold stratified cross-validation on training set
- Validation: Stratified sampling to maintain class balance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
INPUT_FILE = str(Path(__file__).parent.parent / 'data-preprocessing' / 'outputs' / 'data_balanced_3class_scaled.csv')
OUTPUT_DIR = str(Path(__file__).parent.parent / 'test-training-split' / 'outputs')

# Feature columns (based on your preprocessing)
FEATURE_COLUMNS = [
    'LCP_ms', 'FCP_ms', 'TBT_ms', 'CLS', 'TTI_ms', 
    'SpeedIndex_ms', 'TTFB_ms', 'noOfRequests', 
    'pageSize_kb', 'javascriptExecution_ms', 'loadTime_ms'
]

TARGET_COLUMN = 'Performance_Class'

# Split configuration (following Ghattas et al. 2025)
TEST_SIZE = 0.30  # 70% train / 30% test
RANDOM_STATE = 42
N_FOLDS = 10  # For cross-validation

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    print("=" * 80)
    print("TRAIN-TEST SPLIT FOLLOWING GHATTAS ET AL. (2025)")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------
    print(f"\n📂 Loading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    
    print(f"   Total samples: {len(df)}")
    print(f"   Features: {len(FEATURE_COLUMNS)}")
    print(f"   Target: {TARGET_COLUMN}")
    
    # -------------------------------------------------------------------------
    # 2. PREPARE FEATURES AND TARGET
    # -------------------------------------------------------------------------
    print(f"\n🔧 Preparing features and target...")
    
    # Extract features (X) and target (y)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    
    # Verify class distribution
    print(f"\n📊 Original Class Distribution:")
    class_counts = y.value_counts().sort_index()
    for class_name, count in class_counts.items():
        pct = (count / len(y)) * 100
        print(f"   {class_name:<15}: {count:4d} ({pct:5.1f}%)")
    
    # -------------------------------------------------------------------------
    # 3. STRATIFIED TRAIN-TEST SPLIT (70/30)
    # -------------------------------------------------------------------------
    print(f"\n✂️  Performing stratified train-test split (70/30)...")
    print(f"   Following Ghattas et al. (2025) methodology:")
    print(f"   - Test size: {TEST_SIZE * 100:.0f}%")
    print(f"   - Random state: {RANDOM_STATE}")
    print(f"   - Stratified: Yes (maintains class balance)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # CRITICAL: Maintains class balance
    )
    
    print(f"\n✅ Split complete!")
    print(f"   Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Test set:     {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # -------------------------------------------------------------------------
    # 4. VERIFY CLASS BALANCE IN SPLITS
    # -------------------------------------------------------------------------
    print(f"\n📊 Training Set Class Distribution:")
    train_counts = y_train.value_counts().sort_index()
    for class_name, count in train_counts.items():
        pct = (count / len(y_train)) * 100
        print(f"   {class_name:<15}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\n📊 Test Set Class Distribution:")
    test_counts = y_test.value_counts().sort_index()
    for class_name, count in test_counts.items():
        pct = (count / len(y_test)) * 100
        print(f"   {class_name:<15}: {count:4d} ({pct:5.1f}%)")
    
    # -------------------------------------------------------------------------
    # 5. CREATE STRATIFIED K-FOLD CROSS-VALIDATION
    # -------------------------------------------------------------------------
    print(f"\n🔄 Setting up {N_FOLDS}-Fold Stratified Cross-Validation...")
    print(f"   (For use during model training on training set)")
    
    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    
    # Show fold distribution
    print(f"\n   Fold Distribution Preview:")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        val_class_dist = y_train.iloc[val_idx].value_counts().sort_index()
        print(f"   Fold {fold_idx:2d}: Train={len(train_idx):3d}, Val={len(val_idx):3d} | ", end="")
        print(" | ".join([f"{cls}: {count}" for cls, count in val_class_dist.items()]))
        if fold_idx == 3:  # Show first 3 folds only
            print(f"   ... (remaining {N_FOLDS - 3} folds similar)")
            break
    
    # -------------------------------------------------------------------------
    # 6. SAVE SPLITS TO CSV
    # -------------------------------------------------------------------------
    print(f"\n💾 Saving splits to CSV files...")
    
    # Combine features with target for saving
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train
    
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test
    
    # Save
    train_path = Path(OUTPUT_DIR) / 'X_train.csv'
    test_path = Path(OUTPUT_DIR) / 'X_test.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"   ✓ Training set saved: {train_path}")
    print(f"   ✓ Test set saved: {test_path}")
    
    # -------------------------------------------------------------------------
    # 7. SAVE SEPARATE X AND Y (FOR SKLEARN MODELS)
    # -------------------------------------------------------------------------
    print(f"\n💾 Saving separate X and y arrays (for sklearn models)...")
    
    # Save as numpy arrays
    np.save(Path(OUTPUT_DIR) / 'X_train.npy', X_train.values)
    np.save(Path(OUTPUT_DIR) / 'X_test.npy', X_test.values)
    np.save(Path(OUTPUT_DIR) / 'y_train.npy', y_train.values)
    np.save(Path(OUTPUT_DIR) / 'y_test.npy', y_test.values)
    
    print(f"   ✓ X_train.npy, X_test.npy, y_train.npy, y_test.npy saved")
    
    # -------------------------------------------------------------------------
    # 8. SAVE METADATA
    # -------------------------------------------------------------------------
    print(f"\n💾 Saving split metadata...")
    
    metadata = {
        'methodology': 'Ghattas et al. (2025)',
        'total_samples': len(df),
        'n_features': len(FEATURE_COLUMNS),
        'feature_names': FEATURE_COLUMNS,
        'target_column': TARGET_COLUMN,
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'stratified': True,
        'n_folds_cv': N_FOLDS,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'train_class_distribution': train_counts.to_dict(),
        'test_class_distribution': test_counts.to_dict(),
        'class_labels': sorted(y.unique().tolist()),
        'date_created': pd.Timestamp.now().isoformat()
    }
    
    import json
    metadata_path = Path(OUTPUT_DIR) / 'split_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✓ Metadata saved: {metadata_path}")
    
    # -------------------------------------------------------------------------
    # 9. SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("✅ SPLIT COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"""
📊 Dataset Statistics:
   Total samples:        {len(df)}
   Training samples:     {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)
   Test samples:         {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)
   Features:             {len(FEATURE_COLUMNS)}
   Classes:              {len(y.unique())}

🎯 Methodology:
   Paper:                Ghattas et al. (2025)
   Split ratio:          70/30 (training/test)
   Stratification:       Yes (maintains class balance)
   Cross-validation:     {N_FOLDS}-fold stratified
   Random state:         {RANDOM_STATE}

📁 Output Files:
   X_train.csv           Training set (features + target)
   X_test.csv            Test set (features + target)
   X_train.npy           Training features (numpy)
   X_test.npy            Test features (numpy)
   y_train.npy           Training labels (numpy)
   y_test.npy            Test labels (numpy)
   split_metadata.json   Complete metadata

🚀 Next Steps:
   1. Train classifiers using X_train, y_train
   2. Use {N_FOLDS}-fold CV for hyperparameter tuning
   3. Evaluate on X_test, y_test (held-out test set)
   4. Report accuracy, precision, recall, F1-score
""")
    
    print("=" * 80)
    print("✅ Ready for model training!")
    print("=" * 80)


if __name__ == "__main__":
    main()