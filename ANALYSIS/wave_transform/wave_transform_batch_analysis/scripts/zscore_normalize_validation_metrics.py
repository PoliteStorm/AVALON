import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Input CSV from previous aggregation
LATEST_CSV_DIR = sorted(Path('results/validation_metrics_csv').glob('metrics_*'))[-1]
CSV_PATH = LATEST_CSV_DIR / 'validation_metrics.csv'

# Output directories
now = datetime.now().strftime('%Y%m%d_%H%M%S')
Z_OUT_DIR = Path(f'results/validation_metrics_zscore/metrics_{now}')
NORM_OUT_DIR = Path(f'results/validation_metrics_normalized/metrics_{now}')
Z_OUT_DIR.mkdir(parents=True, exist_ok=True)
NORM_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
metrics = pd.read_csv(CSV_PATH)

# Metrics to analyze
metric_cols = ['entropy', 'isi_cv', 'magnitude_cv', 'variance', 'skewness', 'kurtosis', 'n_spikes', 'n_features']

# Z-score analysis
zscore_df = metrics.copy()
for col in metric_cols:
    if col in zscore_df.columns:
        mean = zscore_df[col].mean()
        std = zscore_df[col].std()
        zscore_df[f'{col}_zscore'] = (zscore_df[col] - mean) / std if std > 0 else 0

# Min-max normalization
norm_df = metrics.copy()
for col in metric_cols:
    if col in norm_df.columns:
        minv = norm_df[col].min()
        maxv = norm_df[col].max()
        norm_df[f'{col}_norm'] = (norm_df[col] - minv) / (maxv - minv) if maxv > minv else 0

# Save z-score results
z_csv_path = Z_OUT_DIR / 'validation_metrics_zscore.csv'
z_json_path = Z_OUT_DIR / 'validation_metrics_zscore.json'
zscore_df.to_csv(z_csv_path, index=False)
zscore_df.to_json(z_json_path, orient='records', indent=2)

# Save normalized results
n_csv_path = NORM_OUT_DIR / 'validation_metrics_normalized.csv'
n_json_path = NORM_OUT_DIR / 'validation_metrics_normalized.json'
norm_df.to_csv(n_csv_path, index=False)
norm_df.to_json(n_json_path, orient='records', indent=2)

print(f'✅ Z-score and normalization complete!')
print(f'📄 Z-score CSV: {z_csv_path}')
print(f'📄 Normalized CSV: {n_csv_path}') 