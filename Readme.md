
# **BFRanking (BGP Feature Ranking)**

## **Overview**

This project is a comprehensive tool for analyzing the behavior of Border Gateway Protocol (BGP) anomalies using machine learning models. It builds on the foundation laid by the original repository [bgp-anomaly-classification](https://github.com/thalespaiva/bgp-anomaly-classification) and enhances it with advanced visualization, interpretability, and usability features.

The tool allows for:
- **Model-agnostic SHAP analysis** to interpret feature contributions.
- **Feature importance evaluation**, including global impact and per-class distributions.
- **Subsets analysis** to evaluate model stability across different data samples.
- **Correlation studies** between features and SHAP values.
- A **command-line interface (CLI)** for a streamlined and interactive experience.

## **Features**

- **Feature Impact Analysis:**
  - Global feature importance across all classes.
  - Per-class feature importance visualization.
  - Feature stability across random subsets of data.

- **Advanced Visualizations:**
  - Heatmaps for feature-SHAP value correlations.
  - Summary and waterfall charts to explain individual predictions.
  - Violin plots for distribution comparisons across classes.

- **CLI Support (In Progress):**
  - Load multiple trained models for comparison.
  - Use different testing datasets.
  - Select desired visualizations and analyses dynamically.

- **Extensible Design:**
  - Modular architecture allows for easy integration of new models and analysis types.
  - Ready for scientific research and operational deployment.

## **Getting Started**

### **Installation**

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your constants (features, sequence length, etc.) in the `constants.py` file.

### **Data Preparation**

1. Ensure your data is formatted as NumPy arrays or CSV files:
   - Training data: `x_train.npy` and `y_train.npy`.
   - Validation data: `x_valid.npy`.
   - Testing data: `x_test.npy`.

2. Preprocess the data using the scripts provided in the original repository or adapt them as needed.

### **Usage**

Run the available scripts for different analyses:

#### **Train a Model**
```bash
python train.py --features <path_to_features> --output <output_model_path>
```

#### **SHAP Analysis**
Visualize feature impact:
```bash
python explain.py <path_to_model> <path_to_validation_dataset> <plot_type>
```

`<plot_type>` can be:
- `summary`: Global and per-class feature impact.
- `waterfall`: Waterfall chart for a specific sample.
- `force`: Force plot for a specific sample.
- `impact`: Aggregated feature impact across classes.

#### **Subsets Analysis**
Analyze feature stability across random subsets:
```bash
python subsets.py <path_to_model> <path_to_validation_dataset>
```

#### **Correlation Analysis**
Explore feature correlations with SHAP values:
```bash
python analyze_correlation.py <path_to_model> <path_to_validation_dataset>
```

## **In Progress**

1. **CLI for Unified Analysis:**
   - A single command to load models, datasets, and specify desired analysis or charts.

2. **Multiple Model Comparisons:**
   - Load multiple trained models to compare feature impacts and stability.

3. **Tool Documentation:**
   - Comprehensive CLI usage guide and API documentation for extensibility.

4. **Interactive Visualizations:**
   - Use tools like Plotly or Dash for an interactive user experience.

## **Acknowledgements**

This project is based on the original [bgp-anomaly-classification](https://github.com/thalespaiva/bgp-anomaly-classification) repository, with additional features and enhancements to serve as a robust tool for both research and operational use.
