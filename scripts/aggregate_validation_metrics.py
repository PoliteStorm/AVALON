import os
import json
import csv
from datetime import datetime
from pathlib import Path

# Directory containing result JSONs
RESULTS_DIR = Path('results/ultra_simple_scaling_analysis_improved/json_results')

# Output directories
now = datetime.now().strftime('%Y%m%d_%H%M%S')
CSV_OUT_DIR = Path(f'results/validation_metrics_csv/metrics_{now}')
JSON_OUT_DIR = Path(f'results/validation_metrics_json/metrics_{now}')
CSV_OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Find all result JSON files
json_files = sorted(RESULTS_DIR.glob('ultra_simple_analysis_*.json'))

# Aggregated data
all_metrics = []

for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
    for rate_key, rate_data in data.items():
        # Extract metrics
        stats = rate_data.get('signal_statistics', {})
        spikes = rate_data.get('spike_detection', {})
        sqrt_results = rate_data.get('sqrt_wave_transform_results', {})
        linear_results = rate_data.get('linear_wave_transform_results', {})
        # Use sqrt as primary, fallback to linear if needed
        features = sqrt_results if sqrt_results else linear_results
        # Calculate magnitude CV if possible
        magnitudes = [f['magnitude'] for f in features.get('all_features', [])] if features else []
        magnitude_cv = (float((sum((x - sum(magnitudes)/len(magnitudes))**2 for x in magnitudes)/len(magnitudes))**0.5)/float(sum(magnitudes)/len(magnitudes))) if magnitudes and sum(magnitudes) else None
        # Compose row
        row = {
            'file': json_file.name,
            'rate_key': rate_key,
            'sampling_rate': rate_data.get('sampling_rate'),
            'entropy': features.get('signal_entropy'),
            'isi_cv': spikes.get('isi_cv'),
            'magnitude_cv': magnitude_cv,
            'variance': features.get('signal_variance'),
            'skewness': features.get('signal_skewness'),
            'kurtosis': features.get('signal_kurtosis'),
            'n_spikes': spikes.get('n_spikes'),
            'n_features': features.get('n_features'),
        }
        all_metrics.append(row)

# Write CSV
csv_path = CSV_OUT_DIR / 'validation_metrics.csv'
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = list(all_metrics[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in all_metrics:
        writer.writerow(row)

# Write JSON
json_path = JSON_OUT_DIR / 'validation_metrics.json'
with open(json_path, 'w') as jf:
    json.dump(all_metrics, jf, indent=2)

print(f'✅ Aggregated {len(all_metrics)} validation metric sets.')
print(f'📄 CSV saved to: {csv_path}')
print(f'📄 JSON saved to: {json_path}') 