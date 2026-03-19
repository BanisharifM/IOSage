#!/usr/bin/env python3
"""
Comprehensive Statistical Audit of Normalized Data
===================================================
Checks: integrity, distribution quality, covariate shift,
extreme values, correlation preservation, range/scale.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = "/work/hdd/bdau/mbanisharifdehkordi/SC_2026"

# ── Load data ────────────────────────────────────────────────────────────
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

eng = pd.read_parquet(f"{BASE}/data/processed/production/features.parquet")
train = pd.read_parquet(f"{BASE}/data/processed/splits/train.parquet")
val = pd.read_parquet(f"{BASE}/data/processed/splits/val.parquet")
test = pd.read_parquet(f"{BASE}/data/processed/splits/test.parquet")
eda = pd.read_parquet(f"{BASE}/data/processed/production/eda/stats.parquet")

# Identify feature columns (non-metadata)
info_cols = [c for c in train.columns if c.startswith("_")]
feat_cols = [c for c in train.columns if not c.startswith("_")]

print(f"Pre-normalization (engineered_features): {eng.shape}")
print(f"Train: {train.shape}  Val: {val.shape}  Test: {test.shape}")
print(f"Feature columns: {len(feat_cols)}  Info columns: {len(info_cols)}")
print(f"EDA stats: {eda.shape}")
total_rows = len(train) + len(val) + len(test)
print(f"Total rows across splits: {total_rows} (original: {len(eng)})")

# ── Helpers ───────────────────────────────────────────────────────────────
def section(title):
    print(f"\n{'=' * 80}")
    print(title.upper())
    print("=" * 80)

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  1. BASIC INTEGRITY                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝
section("1. BASIC INTEGRITY")

for name, df in [("Train", train), ("Val", val), ("Test", test)]:
    sub = df[feat_cols]

    # NaN
    nan_counts = sub.isna().sum()
    nan_feats = nan_counts[nan_counts > 0]
    print(f"\n--- {name} ({len(df)} rows) ---")
    print(f"  NaN: {nan_counts.sum()} total across {len(nan_feats)} features")
    if len(nan_feats) > 0:
        print(f"  Top NaN features:")
        for f, c in nan_feats.sort_values(ascending=False).head(20).items():
            print(f"    {f}: {c} ({c/len(df)*100:.1f}%)")

    # Inf
    inf_mask = np.isinf(sub.select_dtypes(include=[np.number]))
    inf_counts = inf_mask.sum()
    inf_feats = inf_counts[inf_counts > 0]
    print(f"  Inf: {inf_counts.sum()} total across {len(inf_feats)} features")
    if len(inf_feats) > 0:
        for f, c in inf_feats.sort_values(ascending=False).head(10).items():
            print(f"    {f}: {c}")

    # Constant features (zero variance)
    numeric_sub = sub.select_dtypes(include=[np.number])
    variances = numeric_sub.var()
    constant_feats = variances[variances == 0].index.tolist()
    print(f"  Constant features (var=0): {len(constant_feats)}")
    if constant_feats:
        for f in constant_feats:
            val_unique = numeric_sub[f].unique()
            print(f"    {f}: unique={val_unique[:5]}")

    # All-zero features
    allzero = (numeric_sub == 0).all()
    allzero_feats = allzero[allzero].index.tolist()
    print(f"  All-zero features: {len(allzero_feats)}")
    if allzero_feats:
        for f in allzero_feats:
            print(f"    {f}")

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  2. DISTRIBUTION QUALITY (Post-Normalization)                        ║
# ╚═══════════════════════════════════════════════════════════════════════╝
section("2. DISTRIBUTION QUALITY (Train Set)")

train_num = train[feat_cols].select_dtypes(include=[np.number])
numeric_feats = train_num.columns.tolist()

dist_stats = pd.DataFrame(index=numeric_feats)
dist_stats["mean"] = train_num.mean()
dist_stats["std"] = train_num.std()
dist_stats["skewness"] = train_num.skew()
dist_stats["kurtosis"] = train_num.kurtosis()  # excess kurtosis
dist_stats["min"] = train_num.min()
dist_stats["max"] = train_num.max()
dist_stats["p01"] = train_num.quantile(0.01)
dist_stats["p99"] = train_num.quantile(0.99)

print(f"\nTotal numeric features: {len(numeric_feats)}")
print(f"\nOverall distribution summary:")
print(f"  Mean of means: {dist_stats['mean'].mean():.4f}")
print(f"  Mean of stds:  {dist_stats['std'].mean():.4f}")
print(f"  Median skew:   {dist_stats['skewness'].median():.4f}")
print(f"  Median kurt:   {dist_stats['kurtosis'].median():.4f}")

# Flag high skewness
high_skew = dist_stats[dist_stats["skewness"].abs() > 10].sort_values("skewness", key=abs, ascending=False)
print(f"\n  Features with |skewness| > 10: {len(high_skew)}")
if len(high_skew) > 0:
    for f, row in high_skew.head(20).iterrows():
        print(f"    {f}: skew={row['skewness']:.2f}, kurt={row['kurtosis']:.2f}")

# Flag high kurtosis
high_kurt = dist_stats[dist_stats["kurtosis"].abs() > 100].sort_values("kurtosis", key=abs, ascending=False)
print(f"\n  Features with |kurtosis| > 100: {len(high_kurt)}")
if len(high_kurt) > 0:
    for f, row in high_kurt.head(20).iterrows():
        print(f"    {f}: kurt={row['kurtosis']:.2f}, skew={row['skewness']:.2f}")

# Flag bad scaling
over_scaled = dist_stats[dist_stats["std"] > 50].sort_values("std", ascending=False)
under_scaled = dist_stats[dist_stats["std"] < 0.001].sort_values("std")
print(f"\n  Features with std > 50 (over-scaled): {len(over_scaled)}")
for f, row in over_scaled.head(10).iterrows():
    print(f"    {f}: std={row['std']:.4f}, mean={row['mean']:.4f}")
print(f"\n  Features with std < 0.001 (under-scaled): {len(under_scaled)}")
for f, row in under_scaled.head(10).iterrows():
    print(f"    {f}: std={row['std']:.6f}, mean={row['mean']:.6f}")

# Compare skewness before vs after normalization
section("2b. SKEWNESS IMPROVEMENT: Before vs After Normalization")

# Get pre-norm stats from EDA (which is pre-normalization)
common_feats = [f for f in eda.index if f in numeric_feats]
print(f"Features with before/after comparison: {len(common_feats)}")

if "skewness" in eda.columns:
    comparison = pd.DataFrame(index=common_feats)
    comparison["skew_before"] = eda.loc[common_feats, "skewness"]
    comparison["skew_after"] = dist_stats.loc[common_feats, "skewness"]
    comparison["improvement"] = comparison["skew_before"].abs() - comparison["skew_after"].abs()

    improved = (comparison["improvement"] > 0).sum()
    worsened = (comparison["improvement"] < 0).sum()
    unchanged = (comparison["improvement"] == 0).sum()
    print(f"\n  Skewness improved: {improved}/{len(common_feats)} ({improved/len(common_feats)*100:.1f}%)")
    print(f"  Skewness worsened: {worsened}/{len(common_feats)} ({worsened/len(common_feats)*100:.1f}%)")
    print(f"  Unchanged:         {unchanged}/{len(common_feats)}")
    print(f"  Mean |skew| before: {comparison['skew_before'].abs().mean():.4f}")
    print(f"  Mean |skew| after:  {comparison['skew_after'].abs().mean():.4f}")

    # Top improvements
    top_improve = comparison.sort_values("improvement", ascending=False).head(10)
    print(f"\n  Top 10 improvements:")
    for f, row in top_improve.iterrows():
        print(f"    {f}: {row['skew_before']:.2f} -> {row['skew_after']:.2f} (delta={row['improvement']:.2f})")

    # Worst regressions
    worst = comparison.sort_values("improvement").head(10)
    print(f"\n  Worst 10 regressions:")
    for f, row in worst.iterrows():
        print(f"    {f}: {row['skew_before']:.2f} -> {row['skew_after']:.2f} (delta={row['improvement']:.2f})")
else:
    print("  EDA stats don't have skewness column — skipping comparison")


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  3. TRAIN-VAL-TEST ALIGNMENT (Covariate Shift)                      ║
# ╚═══════════════════════════════════════════════════════════════════════╝
section("3. COVARIATE SHIFT ANALYSIS")

train_num = train[numeric_feats]
val_num = val[numeric_feats]
test_num = test[numeric_feats]

train_mean = train_num.mean()
val_mean = val_num.mean()
test_mean = test_num.mean()
train_std = train_num.std()

# Mean shift analysis
mean_shift_val = (val_mean - train_mean).abs() / (train_std + 1e-10)
mean_shift_test = (test_mean - train_mean).abs() / (train_std + 1e-10)

shifted_val = mean_shift_val[mean_shift_val > 0.5]
shifted_test = mean_shift_test[mean_shift_test > 0.5]

print(f"\nFeatures with train-val mean shift > 0.5 std: {len(shifted_val)}/{len(numeric_feats)}")
if len(shifted_val) > 0:
    for f in shifted_val.sort_values(ascending=False).head(15).index:
        print(f"  {f}: shift={mean_shift_val[f]:.3f} std "
              f"(train={train_mean[f]:.3f}, val={val_mean[f]:.3f})")

print(f"\nFeatures with train-test mean shift > 0.5 std: {len(shifted_test)}/{len(numeric_feats)}")
if len(shifted_test) > 0:
    for f in shifted_test.sort_values(ascending=False).head(15).index:
        print(f"  {f}: shift={mean_shift_test[f]:.3f} std "
              f"(train={train_mean[f]:.3f}, test={test_mean[f]:.3f})")

# KS test for top shifted features
print(f"\n--- KS Test (top 30 features by mean shift) ---")
top_shift_feats = mean_shift_test.sort_values(ascending=False).head(30).index.tolist()
ks_results = []
for feat in top_shift_feats:
    ks_tv, pv_tv = sp_stats.ks_2samp(train_num[feat].dropna(), val_num[feat].dropna())
    ks_tt, pv_tt = sp_stats.ks_2samp(train_num[feat].dropna(), test_num[feat].dropna())
    ks_results.append({
        "feature": feat,
        "ks_train_val": ks_tv, "pval_train_val": pv_tv,
        "ks_train_test": ks_tt, "pval_train_test": pv_tt,
        "mean_shift_test": mean_shift_test[feat]
    })
ks_df = pd.DataFrame(ks_results)
print(f"  {'Feature':<40s} {'KS(tr-val)':<12s} {'p-val':<12s} {'KS(tr-test)':<12s} {'p-val':<12s} {'MeanShift'}")
for _, row in ks_df.iterrows():
    sig_v = "***" if row["pval_train_val"] < 0.001 else ("**" if row["pval_train_val"] < 0.01 else ("*" if row["pval_train_val"] < 0.05 else ""))
    sig_t = "***" if row["pval_train_test"] < 0.001 else ("**" if row["pval_train_test"] < 0.01 else ("*" if row["pval_train_test"] < 0.05 else ""))
    print(f"  {row['feature']:<40s} {row['ks_train_val']:.4f}{sig_v:<5s}   {row['pval_train_val']:.2e}   "
          f"{row['ks_train_test']:.4f}{sig_t:<5s}   {row['pval_train_test']:.2e}   {row['mean_shift_test']:.3f}")

# Overall summary
print(f"\nOverall shift summary:")
print(f"  Mean absolute shift (train-val):  {mean_shift_val.mean():.4f} std")
print(f"  Mean absolute shift (train-test): {mean_shift_test.mean():.4f} std")
print(f"  Median shift (train-val):  {mean_shift_val.median():.4f} std")
print(f"  Median shift (train-test): {mean_shift_test.median():.4f} std")
n_sig_val = (ks_df["pval_train_val"] < 0.05).sum()
n_sig_test = (ks_df["pval_train_test"] < 0.05).sum()
print(f"  Significant KS (p<0.05) train-val: {n_sig_val}/{len(ks_df)}")
print(f"  Significant KS (p<0.05) train-test: {n_sig_test}/{len(ks_df)}")


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  4. EXTREME VALUES                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════╝
section("4. EXTREME VALUES")

# Max absolute value > 100
max_abs = train_num.abs().max()
extreme_feats = max_abs[max_abs > 100].sort_values(ascending=False)
print(f"\nFeatures with max |value| > 100 (train): {len(extreme_feats)}/{len(numeric_feats)}")
for f, v in extreme_feats.head(30).items():
    print(f"  {f}: max|val|={v:.2f}  (min={train_num[f].min():.2f}, max={train_num[f].max():.2f})")

# Outlier ratio (|z-score from median| > 3)
medians = train_num.median()
mads = (train_num - medians).abs().median()  # MAD
outlier_ratios = {}
for f in numeric_feats:
    if mads[f] > 1e-10:
        z = (train_num[f] - medians[f]).abs() / (mads[f] * 1.4826)  # scaled MAD
        outlier_ratios[f] = (z > 3).mean()
    else:
        # Use std-based if MAD is zero
        if train_std[f] > 1e-10:
            z = (train_num[f] - train_mean[f]).abs() / train_std[f]
            outlier_ratios[f] = (z > 3).mean()
        else:
            outlier_ratios[f] = 0.0

outlier_series = pd.Series(outlier_ratios).sort_values(ascending=False)
high_outlier = outlier_series[outlier_series > 0.05]
print(f"\nFeatures with outlier ratio > 5% (MAD-based z>3): {len(high_outlier)}/{len(numeric_feats)}")
for f, r in high_outlier.head(20).items():
    print(f"  {f}: {r*100:.1f}% outliers  (mean={train_mean[f]:.3f}, std={train_std[f]:.3f}, skew={dist_stats.loc[f,'skewness']:.2f})")

# Overall outlier summary
print(f"\nOutlier ratio distribution:")
print(f"  Mean:   {outlier_series.mean()*100:.2f}%")
print(f"  Median: {outlier_series.median()*100:.2f}%")
print(f"  Max:    {outlier_series.max()*100:.2f}% ({outlier_series.idxmax()})")
print(f"  >1%:    {(outlier_series > 0.01).sum()} features")
print(f"  >5%:    {(outlier_series > 0.05).sum()} features")
print(f"  >10%:   {(outlier_series > 0.10).sum()} features")


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  5. CORRELATION PRESERVATION                                         ║
# ╚═══════════════════════════════════════════════════════════════════════╝
section("5. CORRELATION PRESERVATION (Spearman)")

# Use pre-norm data from engineered_features and post-norm from train
# Align on common features
eng_info = [c for c in eng.columns if c.startswith("_")]
eng_feats = [c for c in eng.columns if not c.startswith("_")]
common = [f for f in eng_feats if f in numeric_feats]

print(f"Common features for correlation comparison: {len(common)}")

# Sample to speed up (50k rows from each)
np.random.seed(42)
n_sample = min(50000, len(eng), len(train))
eng_sample = eng[common].sample(n=n_sample, random_state=42)
train_sample = train_num[common].sample(n=n_sample, random_state=42)

# Compute Spearman correlation matrices
print("Computing pre-norm Spearman correlations...")
corr_before = eng_sample.corr(method="spearman")
print("Computing post-norm Spearman correlations...")
corr_after = train_sample.corr(method="spearman")

# Compare upper triangles
mask = np.triu(np.ones(corr_before.shape, dtype=bool), k=1)
before_vals = corr_before.values[mask]
after_vals = corr_after.values[mask]

# Remove NaN pairs
valid = ~(np.isnan(before_vals) | np.isnan(after_vals))
before_valid = before_vals[valid]
after_valid = after_vals[valid]

print(f"\nCorrelation pairs compared: {valid.sum()}")
diff = np.abs(before_valid - after_valid)
print(f"  Mean |difference|: {diff.mean():.6f}")
print(f"  Max  |difference|: {diff.max():.6f}")
print(f"  Median |diff|:     {np.median(diff):.6f}")
print(f"  Pairs with |diff| > 0.01: {(diff > 0.01).sum()}")
print(f"  Pairs with |diff| > 0.05: {(diff > 0.05).sum()}")
print(f"  Pairs with |diff| > 0.10: {(diff > 0.10).sum()}")

# Correlation between before and after (should be ~1.0)
r_pearson = np.corrcoef(before_valid, after_valid)[0, 1]
r_spearman, _ = sp_stats.spearmanr(before_valid, after_valid)
print(f"\n  Pearson correlation of correlation matrices:  {r_pearson:.6f}")
print(f"  Spearman correlation of correlation matrices: {r_spearman:.6f}")

# Find feature pairs with biggest correlation change
if (diff > 0.05).sum() > 0:
    print(f"\n  Feature pairs with biggest correlation change (|diff| > 0.05):")
    rows, cols = np.where(mask)
    big_changes = []
    for idx in np.argsort(-diff)[:20]:
        r, c = rows[valid][idx], cols[valid][idx]  # This won't work perfectly, let's do it differently
        break
    # Better approach
    diff_matrix = np.abs(corr_after.values - corr_before.values)
    np.fill_diagonal(diff_matrix, 0)
    for _ in range(10):
        idx = np.unravel_index(np.nanargmax(diff_matrix), diff_matrix.shape)
        f1, f2 = common[idx[0]], common[idx[1]]
        b = corr_before.loc[f1, f2]
        a = corr_after.loc[f1, f2]
        print(f"    {f1} <-> {f2}: before={b:.4f}, after={a:.4f}, |diff|={abs(a-b):.4f}")
        diff_matrix[idx[0], idx[1]] = 0
        diff_matrix[idx[1], idx[0]] = 0


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  6. RANGE AND SCALE CHECK                                           ║
# ╚═══════════════════════════════════════════════════════════════════════╝
section("6. RANGE AND SCALE CHECK")

p05 = train_num.quantile(0.05)
p95 = train_num.quantile(0.95)
p25 = train_num.quantile(0.25)
p75 = train_num.quantile(0.75)
iqr = p75 - p25

# Good range: 95th percentile within [-10, 10]
in_good_range = ((p95.abs() <= 10) & (p05.abs() <= 10))
n_good = in_good_range.sum()
print(f"\nFeatures with 95th percentile in [-10, 10]: {n_good}/{len(numeric_feats)} ({n_good/len(numeric_feats)*100:.1f}%)")

bad_range = (~in_good_range)
if bad_range.sum() > 0:
    print(f"\n  Features OUTSIDE good range:")
    for f in p95[bad_range].sort_values(ascending=False).index[:20]:
        print(f"    {f}: p05={p05[f]:.2f}, p95={p95[f]:.2f}, max={train_num[f].max():.2f}")

# Good IQR: between 0.1 and 10
good_iqr = (iqr >= 0.1) & (iqr <= 10)
n_good_iqr = good_iqr.sum()
print(f"\nFeatures with IQR in [0.1, 10]: {n_good_iqr}/{len(numeric_feats)} ({n_good_iqr/len(numeric_feats)*100:.1f}%)")

low_iqr = iqr[iqr < 0.1].sort_values()
high_iqr = iqr[iqr > 10].sort_values(ascending=False)
print(f"\n  Features with IQR < 0.1 (too narrow): {len(low_iqr)}")
for f, v in low_iqr.head(15).items():
    zf = (train_num[f] == 0).mean()
    print(f"    {f}: IQR={v:.4f}, zero_frac={zf:.2f}, std={train_std[f]:.4f}")
print(f"\n  Features with IQR > 10 (too wide): {len(high_iqr)}")
for f, v in high_iqr.head(15).items():
    print(f"    {f}: IQR={v:.2f}, p05={p05[f]:.2f}, p95={p95[f]:.2f}")


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  FINAL SUMMARY                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════╝
section("FINAL SUMMARY — ML READINESS ASSESSMENT")

issues = []
if train[feat_cols].isna().sum().sum() > 0:
    issues.append(f"NaN values present in {(train[feat_cols].isna().sum() > 0).sum()} features")
if np.isinf(train_num).sum().sum() > 0:
    issues.append("Infinite values present")
if len(over_scaled) > 0:
    issues.append(f"{len(over_scaled)} features with std > 50")
if len(under_scaled) > 0:
    issues.append(f"{len(under_scaled)} features with std < 0.001")
if len(high_skew) > 0:
    issues.append(f"{len(high_skew)} features with |skewness| > 10")
if len(high_kurt) > 0:
    issues.append(f"{len(high_kurt)} features with |kurtosis| > 100")
if len(extreme_feats) > 0:
    issues.append(f"{len(extreme_feats)} features with max |value| > 100")
if len(shifted_test) > 0:
    issues.append(f"{len(shifted_test)} features with train-test mean shift > 0.5 std")
if len(high_outlier) > 0:
    issues.append(f"{len(high_outlier)} features with outlier ratio > 5%")

print(f"\n  Total features: {len(numeric_feats)}")
print(f"  Features in good range (95th in [-10,10]): {n_good}/{len(numeric_feats)}")
print(f"  Features with good IQR [0.1, 10]: {n_good_iqr}/{len(numeric_feats)}")
print(f"  Correlation preservation (Pearson r): {r_pearson:.6f}")

if issues:
    print(f"\n  ISSUES FOUND ({len(issues)}):")
    for i, issue in enumerate(issues, 1):
        print(f"    {i}. {issue}")
else:
    print("\n  NO ISSUES FOUND — Data is ML-ready!")

# Severity assessment
print("\n  SEVERITY ASSESSMENT:")
if len(over_scaled) > 0 or len(extreme_feats) > 10:
    print("  [HIGH] Scale issues may cause gradient/convergence problems in neural nets")
if len(high_skew) > 5:
    print("  [MEDIUM] Heavily skewed features — tree models handle this, NNs may struggle")
if len(shifted_test) > len(numeric_feats) * 0.2:
    print("  [HIGH] Significant covariate shift — model may underperform on test set")
elif len(shifted_test) > 0:
    print("  [LOW-MEDIUM] Some covariate shift detected — expected for temporal splits")
if len(high_outlier) > len(numeric_feats) * 0.3:
    print("  [MEDIUM] Many features have high outlier rates — consider clipping")
if r_pearson > 0.999:
    print("  [OK] Correlation structure perfectly preserved")
elif r_pearson > 0.99:
    print("  [OK] Correlation structure well preserved")
else:
    print("  [WARNING] Correlation structure may have changed significantly")

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)
