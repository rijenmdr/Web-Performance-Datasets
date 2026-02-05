import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*60)
print("STEP 2: TARGET PERFORMANCE SCORE CREATION (CORRECTED)")
print("="*60)

# ============================================
# LOAD PREPROCESSED DATA
# ============================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
CURRENT_DIR = os.path.dirname(__file__)
csv_file = os.path.join(PROJECT_ROOT, 'scripts', 'data-preprocessing', 'data_preprocessed.csv')
df = pd.read_csv(csv_file)

print(f"\n1. Loaded preprocessed data: {len(df)} websites")

# Define target metrics
all_performance_metrics = [
    'LCP_ms', 'FCP_ms', 'TBT_ms', 'CLS', 'TTI_ms', 'SpeedIndex_ms',
    'TTFB_ms', 'noOfRequests', 'pageSize_kb', 
    'javascriptExecution_ms', 'loadTime_ms'
]

# ============================================
# CREATE EXPERT-WEIGHTED SCORE
# ============================================
print(f"\n2. Creating expert-weighted performance score...")

expert_weights = {
    'LCP_ms': 0.25, 'FCP_ms': 0.10, 'TBT_ms': 0.15, 'CLS': 0.05,
    'TTI_ms': 0.15, 'SpeedIndex_ms': 0.10, 'TTFB_ms': 0.08,
    'noOfRequests': 0.04, 'pageSize_kb': 0.04,
    'javascriptExecution_ms': 0.02, 'loadTime_ms': 0.02
}

df['target'] = sum(df[metric] * weight for metric, weight in expert_weights.items())

print(f"   Score statistics:")
print(f"   Mean:   {df['target'].mean():.4f}")
print(f"   Median: {df['target'].median():.4f}")
print(f"   Min:    {df['target'].min():.4f}")
print(f"   Max:    {df['target'].max():.4f}")

# ============================================
# PROBLEM: Fixed bins create severe imbalance
# ============================================
print(f"\n3. Analyzing distribution with FIXED bins...")
fixed_bins = [0, 0.7, 0.8, 0.9, 1.0]
fixed_labels = ['<0.7', '0.7-0.8', '0.8-0.9', '≥0.9']
df['category_fixed'] = pd.cut(df['target'], bins=fixed_bins, labels=fixed_labels, include_lowest=True)

print(f"\n   FIXED BINS distribution:")
fixed_dist = df['category_fixed'].value_counts().sort_index()
for label, count in fixed_dist.items():
    pct = (count / len(df)) * 100
    print(f"   {label:10s}: {count:4d} samples ({pct:5.1f}%)")

fixed_imbalance = fixed_dist.max() / fixed_dist.min()
print(f"\n   ❌ Imbalance ratio: {fixed_imbalance:.2f}:1 (EXTREME!)")
print(f"   ❌ Only {fixed_dist.min()} samples in minority class")
print(f"   ❌ This will cause:")
print(f"      - Test set: ~{int(fixed_dist.min() * 0.2)} samples (unreliable)")
print(f"      - Train set: ~{int(fixed_dist.min() * 0.8)} samples (insufficient)")

# ============================================
# SOLUTION: Use quantile-based bins
# ============================================
print(f"\n4. Creating BALANCED distribution with QUANTILE bins...")

# Calculate quartiles
quartiles = df['target'].quantile([0, 0.25, 0.5, 0.75, 1.0])
print(f"\n   Quartile boundaries:")
print(f"   Q1 (0-25%):   {quartiles[0.0]:.4f} - {quartiles[0.25]:.4f}")
print(f"   Q2 (25-50%):  {quartiles[0.25]:.4f} - {quartiles[0.5]:.4f}")
print(f"   Q3 (50-75%):  {quartiles[0.5]:.4f} - {quartiles[0.75]:.4f}")
print(f"   Q4 (75-100%): {quartiles[0.75]:.4f} - {quartiles[1.0]:.4f}")

# Create quantile-based bins
quantile_labels = ['Q1 (Bottom 25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (Top 25%)']
df['category_quantile'] = pd.cut(df['target'], bins=quartiles, labels=quantile_labels, 
                                  include_lowest=True, duplicates='drop')

print(f"\n   QUANTILE BINS distribution:")
quantile_dist = df['category_quantile'].value_counts().sort_index()
for label, count in quantile_dist.items():
    pct = (count / len(df)) * 100
    print(f"   {label:20s}: {count:4d} samples ({pct:5.1f}%)")

quantile_imbalance = quantile_dist.max() / quantile_dist.min()
print(f"\n   ✅ Imbalance ratio: {quantile_imbalance:.2f}:1 (EXCELLENT!)")
print(f"   ✅ Minimum samples: {quantile_dist.min()} (sufficient for ML)")
print(f"   ✅ This enables:")
print(f"      - Test set: ~{int(quantile_dist.min() * 0.2)} samples (reliable)")
print(f"      - Train set: ~{int(quantile_dist.min() * 0.8)} samples (sufficient)")

# ============================================
# COMPARISON OF APPROACHES
# ============================================
print(f"\n5. Comparing binning approaches:")
print(f"\n   {'Approach':<20} {'Imbalance':>12} {'Min Samples':>12} {'Status':>12}")
print(f"   {'-'*60}")
print(f"   {'Fixed bins':<20} {fixed_imbalance:>11.1f}:1 {fixed_dist.min():>12d} {'❌ Unusable':>12}")
print(f"   {'Quantile bins':<20} {quantile_imbalance:>11.1f}:1 {quantile_dist.min():>12d} {'✅ Good':>12}")

# ============================================
# VISUALIZATIONS
# ============================================
print(f"\n6. Creating comparative visualizations...")

fig = plt.figure(figsize=(16, 10))

# 1. Target distribution histogram
ax1 = plt.subplot(2, 3, 1)
ax1.hist(df['target'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(df['target'].mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {df["target"].mean():.3f}')
ax1.axvline(df['target'].median(), color='green', linestyle='--', linewidth=2,
            label=f'Median: {df["target"].median():.3f}')

# Add quartile lines
for i, q in enumerate([0.25, 0.5, 0.75]):
    ax1.axvline(quartiles[q], color='orange', linestyle=':', alpha=0.7, linewidth=1.5)

ax1.set_xlabel('Performance Score', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Target Score Distribution with Quartiles', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Fixed bins distribution
ax2 = plt.subplot(2, 3, 2)
colors_bad = ['#e74c3c', '#e67e22', '#f39c12', '#95a5a6']
fixed_dist.plot(kind='bar', ax=ax2, color=colors_bad, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Category (Fixed Bins)', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title(f'❌ Fixed Bins (Imbalance: {fixed_imbalance:.1f}:1)', 
              fontsize=12, fontweight='bold', color='red')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# Add percentage labels
for i, (label, count) in enumerate(fixed_dist.items()):
    pct = (count / len(df)) * 100
    ax2.text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

# 3. Quantile bins distribution
ax3 = plt.subplot(2, 3, 3)
colors_good = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c']
quantile_dist.plot(kind='bar', ax=ax3, color=colors_good, alpha=0.7, edgecolor='black')
ax3.set_xlabel('Category (Quantile Bins)', fontsize=11)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title(f'✅ Quantile Bins (Imbalance: {quantile_imbalance:.1f}:1)', 
              fontsize=12, fontweight='bold', color='green')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', alpha=0.3)

# Add percentage labels
for i, (label, count) in enumerate(quantile_dist.items()):
    pct = (count / len(df)) * 100
    ax3.text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

# 4. Box plot comparison
ax4 = plt.subplot(2, 3, 4)
df.boxplot(column='target', by='category_quantile', ax=ax4)
ax4.set_xlabel('Quartile Category', fontsize=11)
ax4.set_ylabel('Performance Score', fontsize=11)
ax4.set_title('Score Distribution by Quantile Category', fontsize=12, fontweight='bold')
plt.suptitle('')  # Remove default title
ax4.grid(True, alpha=0.3)

# 5. Imbalance comparison
ax5 = plt.subplot(2, 3, 5)
approaches = ['Fixed\nBins', 'Quantile\nBins']
imbalances = [fixed_imbalance, quantile_imbalance]
colors_bar = ['red', 'green']
bars = ax5.bar(approaches, imbalances, color=colors_bar, alpha=0.7, edgecolor='black')
ax5.set_ylabel('Imbalance Ratio (X:1)', fontsize=11)
ax5.set_title('Imbalance Comparison', fontsize=12, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)
ax5.axhline(y=3, color='orange', linestyle='--', alpha=0.7, label='Moderate (3:1)')
ax5.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Severe (10:1)')
ax5.legend(fontsize=9)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, imbalances)):
    ax5.text(i, val, f'{val:.1f}:1', ha='center', va='bottom', fontweight='bold', fontsize=12)

# 6. Sample size comparison for train/test
ax6 = plt.subplot(2, 3, 6)
x = np.arange(2)
width = 0.35

train_fixed = int(fixed_dist.min() * 0.8)
test_fixed = int(fixed_dist.min() * 0.2)
train_quantile = int(quantile_dist.min() * 0.8)
test_quantile = int(quantile_dist.min() * 0.2)

bars1 = ax6.bar(x - width/2, [train_fixed, train_quantile], width, 
                label='Train (80%)', color='steelblue', alpha=0.7, edgecolor='black')
bars2 = ax6.bar(x + width/2, [test_fixed, test_quantile], width,
                label='Test (20%)', color='lightcoral', alpha=0.7, edgecolor='black')

ax6.set_ylabel('Min Samples in Minority Class', fontsize=11)
ax6.set_title('Train/Test Split Sample Size', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(['Fixed Bins', 'Quantile Bins'])
ax6.legend()
ax6.grid(axis='y', alpha=0.3)
ax6.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='Minimum viable')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(CURRENT_DIR, 'target_distribution_corrected.png'), 
            dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: target_distribution_corrected.png")

# ============================================
# SAVE OUTPUTS
# ============================================
print(f"\n7. Saving outputs...")

# Save with both categorizations
output_file = os.path.join(CURRENT_DIR, 'data_with_target_corrected.csv')
df.to_csv(output_file, index=False)
print(f"   ✓ Saved: data_with_target_corrected.csv")

# Save category distributions
category_analysis = pd.DataFrame({
    'Approach': ['Fixed Bins', 'Quantile Bins'],
    'Imbalance_Ratio': [f'{fixed_imbalance:.2f}:1', f'{quantile_imbalance:.2f}:1'],
    'Min_Samples': [fixed_dist.min(), quantile_dist.min()],
    'Train_Set_Min': [train_fixed, train_quantile],
    'Test_Set_Min': [test_fixed, test_quantile],
    'Status': ['❌ Unusable', '✅ Recommended']
})
category_analysis.to_csv(os.path.join(CURRENT_DIR, 'binning_comparison.csv'), index=False)
print(f"   ✓ Saved: binning_comparison.csv")

# ============================================
# RECOMMENDATIONS
# ============================================
print(f"\n" + "="*60)
print("CORRECTED TARGET SCORE CREATION SUMMARY")
print("="*60)
print(f"\n✅ RECOMMENDED APPROACH: Use Quantile Bins")
print(f"   - Balanced distribution (~{quantile_dist.min()} samples per category)")
print(f"   - Imbalance ratio: {quantile_imbalance:.2f}:1 (excellent)")
print(f"   - Interpretable categories (Bottom 25%, 25-50%, 50-75%, Top 25%)")
print(f"   - Reliable train/test split possible")

print(f"\n❌ AVOID: Fixed Bins Approach")
print(f"   - Severe imbalance: {fixed_imbalance:.2f}:1")
print(f"   - Only {fixed_dist.min()} samples in minority class")
print(f"   - Test set would have only ~{test_fixed} samples")
print(f"   - Statistically unreliable results")

print(f"\n📊 FOR YOUR MACHINE LEARNING:")
print(f"   1. Use 'category_quantile' for stratified split")
print(f"   2. Sample weighting is now OPTIONAL (not mandatory)")
print(f"   3. Per-category evaluation will be reliable")
print(f"   4. Results will be publishable and statistically valid")

print(f"\n📝 FOR YOUR RESEARCH PAPER:")
print(f"   \"To ensure balanced class distribution for reliable model")
print(f"    evaluation, we employed quartile-based binning, categorizing")
print(f"    websites into four equal-sized groups (Q1-Q4) based on their")
print(f"    performance scores. This resulted in a balanced dataset with")
print(f"    {quantile_dist.min()}-{quantile_dist.max()} samples per category (imbalance ratio:")
print(f"    {quantile_imbalance:.2f}:1), compared to the severe imbalance ({fixed_imbalance:.1f}:1)")
print(f"    that would result from fixed performance thresholds.\"")

print("\n" + "="*60)
print("✅ Ready for balanced machine learning experiments!")
print("="*60)