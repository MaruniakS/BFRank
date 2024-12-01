import shap
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from constants import CLASS_LABELS, FEATURES, SEQUENCE_LENGTH

# Set seeds for reproducibility
from numpy.random import seed
seed(1)
tf.random.set_seed(2)

def compute_global_feature_impact(shap_values):
    """
    Aggregate SHAP values across all samples, time steps, and classes.
    Returns a dictionary of global feature importance.
    """
    shap_values = np.array(shap_values)  # Convert to NumPy array if needed
    aggregated_shap_values = np.mean(np.abs(shap_values), axis=(0, 1, 2))  # Global aggregation
    return aggregated_shap_values

def compute_per_class_feature_impact(shap_values):
    """
    Compute feature importance separately for each class.
    Returns a dictionary of feature importance per class.
    """
    shap_values = np.array(shap_values)
    per_class_impact = {}
    for class_idx in range(shap_values.shape[0]):
        per_class_impact[class_idx] = np.mean(np.abs(shap_values[class_idx]), axis=(0, 1))
    return per_class_impact

def visualize_feature_impact(aggregated_shap_values, feature_names, per_class_impact=None):
    """
    Visualize global and per-class feature impact.
    """
    # Global feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(feature_names, aggregated_shap_values)
    plt.title("Global Feature Impact (All Classes)")
    plt.xlabel("Features")
    plt.ylabel("Mean Absolute SHAP Value")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.show()

    # Per-class feature importance
    if per_class_impact:
        for class_idx, class_impact in per_class_impact.items():
            plt.figure(figsize=(12, 6))
            plt.bar(feature_names, class_impact)
            plt.title(f"Feature Impact for {CLASS_LABELS[class_idx]} Annomalies")
            plt.xlabel("Features")
            plt.ylabel("Mean Absolute SHAP Value")
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis="y")
            plt.show()

def explain_model_features(model, x_data, feature_names, sequence_length):
    """
    Compute SHAP values and analyze feature impact on the model results.
    """
    n_features = len(feature_names)
    x_data_flattened = x_data.reshape(x_data.shape[0], -1)  # Flatten input for KernelExplainer

    # Define background and test data
    background_data = x_data_flattened[:10]  # Background for SHAP
    test_data = x_data_flattened[:100]  # Analyze first 100 samples

    # Initialize SHAP KernelExplainer
    explainer = shap.KernelExplainer(
        lambda x: model.predict(x.reshape(-1, sequence_length, n_features)),
        background_data
    )

    # Compute SHAP values for test data
    shap_values = explainer.shap_values(test_data)
    shap_values_reshaped = [
        values.reshape(-1, sequence_length, n_features) for values in shap_values
    ]

    # Aggregate results
    global_impact = compute_global_feature_impact(shap_values_reshaped)
    per_class_impact = compute_per_class_feature_impact(shap_values_reshaped)

    # Visualize results
    visualize_feature_impact(global_impact, feature_names, per_class_impact)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to the trained model.')
    parser.add_argument('features', help='Path to the dataset file containing features and labels.')
    args = parser.parse_args()

    # Load data
    x_valid = np.load('x_valid.npy')  # Adjust this as needed for your input data

    # Load the trained model
    print("Loading model...")
    model = tf.keras.models.load_model(args.model)

    # Explain model features
    explain_model_features(model, x_valid, FEATURES, SEQUENCE_LENGTH)

if __name__ == "__main__":
    main()
