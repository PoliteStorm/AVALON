import os
import csv
import re
import sys
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

def get_file_size(path):
    size = os.path.getsize(path)
    if size > 1024**2:
        return f"{size/1024**2:.1f} MB"
    else:
        return f"{size/1024:.1f} KB"

def count_lines(path, max_lines=100000):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, _ in enumerate(f, 1):
                if i > max_lines:
                    return f">{max_lines}"
            return i
    except Exception:
        return 'unknown'

def read_preview(path, n=10):
    lines = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(n):
                line = f.readline()
                if not line:
                    break
                lines.append(line.strip())
    except Exception:
        lines.append('[Error reading file]')
    return lines

def detect_header_and_columns(path):
    preview = read_preview(path, 2)
    if not preview:
        return 'empty', []
    first = preview[0]
    if ',' in first and not re.match(r'^[\d\-\.]+(,[\d\-\.]+)+$', first):
        # Looks like a header
        columns = [c.strip() for c in first.split(',')]
        return 'header', columns
    else:
        return 'no header', []

def parse_coordinate_filename(filename):
    # Example: Pv_L_I+4xR_Fc_N_36d_1_coordinates.csv
    m = re.match(r'([A-Za-z]+)_([A-Za-z0-9\+x]+)_([A-Za-z]+)_([A-Za-z]+)_([\d]+[dh])_([\d]+)_coordinates', filename)
    if m:
        return {
            'species': m.group(1),
            'variant': m.group(2),
            'unknown1': m.group(3),
            'unknown2': m.group(4),
            'duration': m.group(5),
            'replicate': m.group(6)
        }
    else:
        return {}

def profile_csvs(root_dirs):
    summary = []
    for root in root_dirs:
        for fname in os.listdir(root):
            if not fname.endswith('.csv'):
                continue
            path = os.path.join(root, fname)
            size = get_file_size(path)
            n_lines = count_lines(path, 100000)
            preview = read_preview(path, 10)
            header_type, columns = detect_header_and_columns(path)
            meta = {}
            if root.endswith('csv_data'):
                meta = parse_coordinate_filename(fname)
            summary.append({
                'file': fname,
                'directory': root,
                'size': size,
                'lines': n_lines,
                'header_type': header_type,
                'columns': '; '.join(columns) if columns else '',
                'preview': ' | '.join(preview),
                **meta
            })
    return summary

def write_csv(summary, out_path):
    keys = summary[0].keys()
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

def write_md(summary, out_path):
    keys = summary[0].keys()
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('| ' + ' | '.join(keys) + ' |\n')
        f.write('| ' + ' | '.join(['---']*len(keys)) + ' |\n')
        for row in summary:
            f.write('| ' + ' | '.join(str(row[k]) for k in keys) + ' |\n')

def main():
    root_dirs = ['15061491', 'csv_data']
    print(f"Profiling CSVs in: {root_dirs}")
    summary = profile_csvs(root_dirs)
    print(f"Profiled {len(summary)} CSV files.")
    write_csv(summary, 'csv_profile_summary.csv')
    write_md(summary, 'csv_profile_summary.md')
    print("\nSample summary:")
    for row in summary[:5]:
        print(row)
    print("\nFull summary written to csv_profile_summary.csv and csv_profile_summary.md")

if __name__ == '__main__':
    main() 