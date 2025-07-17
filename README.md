# Wave Transform Batch Analysis System

**Version:** 2.0 (Centralized Configuration)
**Last Updated:** 2025-07-16
**Status:** Production Ready

---

## 🗂️ Directory Overview

See `DIRECTORY_OVERVIEW.md` for a full, up-to-date directory map and explanation of all folders and files.

---

## 🎯 Overview

This system provides comprehensive wave transform analysis of fungal electrical activity using Adamatzky's 2023 validated parameters. The system eliminates forced parameters through centralized configuration management and provides dynamic adaptation based on data characteristics.

### Key Features
- **Centralized Configuration:** No forced parameters across all scripts
- **Dynamic Compression:** Adaptive compression based on data length
- **Adamatzky Compliance:** Strict adherence to validated biological parameters
- **Comprehensive Validation:** Multi-layered validation framework
- **Professional Documentation:** Complete process transparency

---

## 🚀 Quick Start

### 1. Show Adamatzky's Methods
```bash
python3 scripts/show_adamatzky_methods.py
```

### 2. Compare Your Results to Standards
```bash
python3 scripts/compare_results.py
```

### 3. Fun, Rigorous Review
```bash
python3 scripts/fun_scientific_review.py
```

### 4. Run the Full Analysis
```bash
python3 scripts/run_centralized_analysis.py
```

---

## ⚙️ Configuration System

### **No Forced Parameters**
All parameters are managed through the centralized configuration system:

```python
from config.analysis_config import config
# Get parameters
dynamic_compression = config.get_compression_factor(data_length)
validation_thresholds = config.get_validation_thresholds()
```

### **Dynamic Parameter Adjustment**
- **Compression:** Automatically adjusted based on data length
- **Thresholds:** Configurable validation thresholds
- **Wave Transform:** Parameter ranges from configuration
- **Validation:** Adaptive validation criteria

---

## 📈 Analysis Pipeline

### **1. Data Preprocessing**
- **File Identification:** Adamatzky-compliant CSV detection
- **Dynamic Compression:** Adaptive compression based on data length
- **Quality Assessment:** Signal quality validation

### **2. Wave Transform Analysis**
- **Formula:** W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
- **Parameters:** k and τ ranges from configuration
- **Feature Detection:** Magnitude threshold from config

### **3. Validation Framework**
- **Biological Validation:** Adamatzky temporal scale compliance
- **Mathematical Validation:** Energy conservation, orthogonality
- **False Positive Detection:** Uniformity and randomness tests

### **4. Visualization and Reporting**
- **Adamatzky Plots:** Standardized analysis plots
- **Wave Transform Heatmaps:** k vs τ visualizations
- **Comprehensive Reports:** JSON and Markdown outputs

---

## 📚 Documentation

- **Directory Overview:** `DIRECTORY_OVERVIEW.md`
- **Process Documentation:** `docs/process_documentation.md`
- **Improvement Strategies:** `docs/improvement_strategies.md`
- **API Documentation:** `docs/api/`
- **User Guides:** `docs/user_guide/`

---

## 🎉 Benefits

- **No Forced Parameters:** All parameters configurable
- **Dynamic Adaptation:** System adjusts to data
- **Validation Framework:** Quality assurance built-in
- **Reproducible Results:** Centralized configuration
- **Clear Structure:** Logical organization
- **Complete Documentation:** Process transparency
- **Modular Design:** Easy maintenance and updates
- **Fun Interface:** Engaging but scientifically rigorous

---

*This system ensures scientific rigor, professional standards, and delightful user experience while maintaining complete transparency and eliminating forced parameters.* 