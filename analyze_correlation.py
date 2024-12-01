import shap
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from constants import CLASS_LABELS, FEATURES, SEQUENCE_LENGTH

from numpy.random import seed
seed(1)
tf.random.set_seed(2)

def compute_feature_shap_correlation(x_data, shap_values, feature_names):
    """
    Compute correlation between feature values and their SHAP values.

    Parameters:
        x_data: Input data (n_samples, sequence_length, n_features).
        shap_values: SHAP values (n_classes, n_samples, sequence_length, n_features).
        feature_names: List of feature names.

    Returns:
        DataFrame containing correlation coefficients for each feature.
    """
    n_classes = shap_values.shape[0]
    shap_values_flattened = np.mean(np.abs(shap_values), axis=2)  # Aggregate over time steps
    correlations = {}

    for class_idx in range(n_classes):
        class_correlations = []
        for feature_idx, feature_name in enumerate(feature_names):
            # Flatten both feature values and SHAP values
            feature_values = x_data[:, :, feature_idx].flatten()[:shap_values_flattened[class_idx].shape[0]]
            shap_feature_values = shap_values_flattened[class_idx, :, feature_idx]

            # Check if the lengths match
            if len(feature_values) != len(shap_feature_values):
                raise ValueError(
                    f"Length mismatch: feature_values={len(feature_values)}, "
                    f"shap_feature_values={len(shap_feature_values)}"
                )

            # Compute correlation
            correlation = np.corrcoef(feature_values, shap_feature_values)[0, 1]
            class_correlations.append(correlation)
        correlations[f"{CLASS_LABELS[class_idx]}"] = class_correlations

    return pd.DataFrame(correlations, index=feature_names)


def visualize_feature_shap_correlation(correlation_df):
    """
    Visualize feature-wise SHAP value correlations as a heatmap.

    Parameters:
        correlation_df: DataFrame with feature-wise SHAP value correlations for each class.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation with SHAP Values")
    plt.xlabel("Classes")
    plt.ylabel("Features")
    plt.xticks(rotation=45, ha="right")
    plt.show()

def explain_with_shap_correlation(model, x_data, feature_names, sequence_length):
    n_features = len(feature_names)
    x_data_flattened = x_data.reshape(x_data.shape[0], -1)

    background_data = x_data_flattened[:10]
    test_data = x_data_flattened[:100]  # Increase sample size for robust correlation

    explainer = shap.KernelExplainer(
        lambda x: model.predict(x.reshape(-1, sequence_length, n_features)),
        background_data
    )

    shap_values = explainer.shap_values(test_data)
    shap_values_reshaped = [
        values.reshape(-1, sequence_length, n_features) for values in shap_values
    ]

    # Compute and visualize feature correlations
    correlation_df = compute_feature_shap_correlation(x_data, np.array(shap_values_reshaped), feature_names)
    visualize_feature_shap_correlation(correlation_df)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to the trained model.')
    parser.add_argument('features', help='Path to the dataset file containing features and labels.')
    args = parser.parse_args()

    x_valid = np.load('x_valid.npy')
    print("Loading model for SHAP explanations...")
    model = tf.keras.models.load_model(args.model)

    explain_with_shap_correlation(model, x_valid, FEATURES, SEQUENCE_LENGTH)

if __name__ == "__main__":
    main()
