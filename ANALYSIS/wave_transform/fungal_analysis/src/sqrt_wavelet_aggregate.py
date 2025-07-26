from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import datetime

RESULTS_ROOT = Path("/home/kronos/AVALON/fungal_analysis/sqrt_wavelet_results")
OUTPUT_DIR = Path("/home/kronos/AVALON/fungal_analysis/sqrt_wavelet_aggregate")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _collect_magnitude_files(root: Path) -> List[Path]:
    """Return all magnitude.npy paths under *root*."""
    return sorted(root.rglob("magnitude.npy"))


def _compute_band_spectra(mag: np.ndarray) -> Dict[str, np.ndarray]:
    """Return τ- and k-spectra (mean over other axis)."""
    tau_spec = mag.mean(axis=1)  # shape (len(τ),)
    k_spec = mag.mean(axis=0)    # shape (len(k),)
    return {"tau": tau_spec, "k": k_spec}


def _parse_label(file_path: Path) -> str:
    """Basic label derived from parent directory name (file stem)."""
    return file_path.parent.stem  # directory named after recording


def build_band_feature_dataframe(mag_paths: List[Path]) -> pd.DataFrame:
    """Create DataFrame with spectra features for every recording."""
    rows = []
    for p in mag_paths:
        mag = np.load(p)
        spectra = _compute_band_spectra(mag)
        label = _parse_label(p)

        # Flatten spectra into single row
        row = {"recording": label}
        row.update({f"tau_{i}": v for i, v in enumerate(spectra["tau"])})
        row.update({f"k_{j}": v for j, v in enumerate(spectra["k"])})
        rows.append(row)

    df = pd.DataFrame(rows).set_index("recording")
    return df


def plot_ridgeline(df: pd.DataFrame, prefix: str):
    """Plot overlaid τ- and k- spectra."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Tau spectra
    tau_cols = [c for c in df.columns if c.startswith("tau_")]
    tau_x = np.arange(len(tau_cols))
    plt.figure(figsize=(10, 6))
    for rec in df.index:
        plt.plot(tau_x, df.loc[rec, tau_cols].values, alpha=0.4, label=rec)
    plt.title("τ-band Collapsed Spectra (mean over k)")
    plt.xlabel("τ index")
    plt.ylabel("Mean |W|")
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{prefix}_tau_spectra_{timestamp}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    # k spectra
    k_cols = [c for c in df.columns if c.startswith("k_")]
    k_x = np.arange(len(k_cols))
    plt.figure(figsize=(10, 6))
    for rec in df.index:
        plt.plot(k_x, df.loc[rec, k_cols].values, alpha=0.4, label=rec)
    plt.title("k-band Collapsed Spectra (mean over τ)")
    plt.xlabel("k index")
    plt.ylabel("Mean |W|")
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{prefix}_k_spectra_{timestamp}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def build_fingerprint_matrix(mag_paths: List[Path], downsample: int = 8) -> pd.DataFrame:
    """Flatten magnitude matrices (optionally downsample) -> DataFrame."""
    flat_rows = []
    labels = []
    for p in mag_paths:
        mag = np.load(p)
        # Downsample with simple slicing if desired
        if downsample > 1:
            mag_ds = mag[::downsample, ::downsample]
        else:
            mag_ds = mag
        flat_rows.append(mag_ds.flatten())
        labels.append(_parse_label(p))

    X = np.vstack(flat_rows)
    df = pd.DataFrame(X, index=labels)
    return df


def plot_pca_fingerprint(matrix_df: pd.DataFrame, prefix: str):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    X = matrix_df.values
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X)
    comp_df = pd.DataFrame(components, columns=["PC1", "PC2"], index=matrix_df.index)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="PC1", y="PC2", data=comp_df, hue=comp_df.index, legend=False)
    plt.title("Fingerprint Matrix – PCA Scatter")
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{prefix}_pca_{timestamp}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    # Save components for later
    comp_df.to_csv(OUTPUT_DIR / f"{prefix}_pca_components_{timestamp}.csv")

    # Cluster heat-map
    sns.clustermap(matrix_df, z_score=0, cmap="mako", figsize=(10, 10))
    plt.title("Fingerprint Matrix – Clustermap (z-scored)")
    out_path = OUTPUT_DIR / f"{prefix}_clustermap_{timestamp}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    mag_files = _collect_magnitude_files(RESULTS_ROOT)
    if not mag_files:
        print("No magnitude.npy files found. Run sqrt_wavelet_scanner first.")
        return

    print(f"Found {len(mag_files)} magnitude files. Building band spectra…")

    # Band-collapsed spectra
    band_df = build_band_feature_dataframe(mag_files)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    band_df.to_csv(OUTPUT_DIR / f"sqrt_wavelet_band_features_{timestamp}.csv")
    plot_ridgeline(band_df, prefix="band")

    # Fingerprint matrix + PCA + clustering
    print("Building fingerprint matrix…")
    fp_df = build_fingerprint_matrix(mag_files, downsample=8)  # 32×8=4 (approx) dims vs 32×64 if default
    plot_pca_fingerprint(fp_df, prefix="fp")

    print("Aggregation complete. Outputs saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main() 